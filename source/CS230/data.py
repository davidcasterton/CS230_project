import copy
import logging
import numpy
import os
import pandas
import pdb
import plotly
import pprint
import pyarrow
import pyarrow.parquet as pq
import six
import sys
import time

from . import common

COLUMNS_ORIG = ['time', 'handwheelAngle', 'throttle', 'brake', 'altitude', 'horizontalSpeed', 'vxCG', 'vyCG', 'yawAngle', 'pitchAngle', 'rollAngle', 'distance']  # 'latitude', 'longitude'
COLUMNS_TO_DIFF = ['yawAngle', 'pitchAngle', 'rollAngle', 'horizontalSpeed', 'distance', 'vxCG', 'vyCG']
COLUMN_DIFF_PREFIX = 'diff_'

COLUMNS = copy.deepcopy(COLUMNS_ORIG)
for column in COLUMNS_TO_DIFF:
    new_column = COLUMN_DIFF_PREFIX + column
    COLUMNS.append(new_column)

COLUMNS_WITH_GPS_JUMP = ['horizontalSpeed', 'vxCG', 'vyCG', 'yawAngle', 'pitchAngle', 'rollAngle']
DIFF_COLUMNS_WITH_GPS_JUMP = [COLUMN_DIFF_PREFIX + x for x in COLUMNS_WITH_GPS_JUMP]

DEFAULT_THRESHOLDS = [
    ('diff_yawAngle', 10),
    ('diff_pitchAngle', 2),
    ('diff_rollAngle', 2),
    ('diff_distance', 10),
    ('diff_vxCG', 10),
    ('diff_vyCG', 10)
]


def load(file_path):
    # read parquet
    table = pq.read_table(file_path)

    # convert parquet table to pandas dataframe
    df = table.to_pandas()

    return df


def add_diffs(df, stride, columns_to_diff=COLUMNS_TO_DIFF):
    logger = logging.getLogger(common.LOG_ROOT)

    for column in columns_to_diff:
        new_column = COLUMN_DIFF_PREFIX + column

        df[new_column] = df[column] - df[column].shift(stride)

    # correct for yawAngle sign flip
    indexes = df.index[(df['diff_yawAngle'] > 300) | (df['diff_yawAngle'] < -300)]
    for i in indexes:
        df.at[i, 'diff_yawAngle'] = (180 - abs(df.iloc[i]['yawAngle'])) + (180 - abs(df.iloc[i - stride]['yawAngle']))

    logger.warning('# diff_yawAngle fixed: %s' % len(indexes))

    # replace NaN with zeros in diff columns
    values = {COLUMN_DIFF_PREFIX + x: 0 for x in columns_to_diff}
    df.fillna(value=values, inplace=True)

    return df


def clean_discontinuities(df, stride, thresholds=DEFAULT_THRESHOLDS):
    logger = logging.getLogger(common.LOG_ROOT)

    for column, threshold in thresholds:
        # find indexes outside of threshold
        indexes = df.index[(df[column] > threshold) | (df[column] < -threshold)]

        for i in indexes:
            if numpy.isnan(df.iloc[i - stride]['altitude']):
                logger.warning('GPS NaN at %s : %s -> %s', i,
                        df.iloc[i][DIFF_COLUMNS_WITH_GPS_JUMP].to_string(header=False, index=False).replace(os.linesep, ','),
                        df.iloc[i - 1][DIFF_COLUMNS_WITH_GPS_JUMP].to_string(header=False, index=False).replace(os.linesep, ','))

                df.at[i, DIFF_COLUMNS_WITH_GPS_JUMP] = df.iloc[i - stride][DIFF_COLUMNS_WITH_GPS_JUMP]

    return df


def display_discontinuities(df, stride, thresholds=DEFAULT_THRESHOLDS):
    logger = logging.getLogger(common.LOG_ROOT)

    for column, index, _df in discontinuity_generator(df):
        logger.debug('%s : %s', column, index)
        display(_df)


def discontinuity_generator(df, thresholds=DEFAULT_THRESHOLDS):
    for column, threshold in thresholds:
        indexes = df.index[(df[column] > threshold) | (df[column] < -threshold)]

        for index in indexes:
            yield column, index, df.iloc[index - 1: index + 2, :]


def get_plotly_fig(df, file_path, stride=2500):
    data = []
    times = df['time'].values

    for sig_name in COLUMNS_TO_DIFF:
        trace = plotly.graph_objs.Scatter(
            x=times[0::stride],
            y=df[COLUMN_DIFF_PREFIX + sig_name].values[0::stride],
            name=COLUMN_DIFF_PREFIX + sig_name,
        )

        data.append(trace)

    layout = plotly.graph_objs.Layout(
        title='/'.join(file_path.split('/')[-2:])
    )

    fig = plotly.graph_objs.Figure(data=data, layout=layout)

    return fig

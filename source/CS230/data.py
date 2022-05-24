import copy
import hashlib
import logging
import math
import numpy
import os
import pandas
import pdb
import plotly
import pprint
import pyarrow
import pyarrow.parquet as pq
import random
import six
import sys
import time

from . import common

DATA_DIR = os.path.join(os.path.abspath(__file__).split('CS230_project')[0], 'CS230_project', 'data')

COLUMNS_ALL = ['HDOP', 'PDOP', 'PPS', 'altitude', 'axCG', 'ayCG', 'azCG', 'brake', 'chassisAccelFL', 'chassisAccelFR',
                'chassisAccelRL', 'chassisAccelRR', 'clutch', 'deflectionFL', 'deflectionFR',
                'distance', 'engineSpeed', 'gpsOrientMode', 'gpsPosMode', 'gpsTime', 'gpsVelMode',
                'handwheelAngle', 'horizontalSpeed', 'latitude', 'longitude', 'numSVsTracked', 'orientAccuracy_heading',
                'orientAccuracy_pitch', 'orientAccuracy_roll', 'pitchAngle', 'pitchRate', 'posAccuracy_down',
                'posAccuracy_east', 'posAccuracy_north', 'rollAngle', 'rollRate', 'sideSlip', 'throttle', 'time',
                'vEast', 'vNorth', 'vUp', 'velAccuracy_down', 'velAccuracy_east', 'velAccuracy_north', 'vxCG', 'vyCG',
                'vzCG', 'wheelAccelFL', 'wheelAccelFR', 'wheelAccelRL', 'wheelAccelRR', 'yawAngle', 'yawRate']
COLUMNS_HUMAN_INPUT = ['brake', 'throttle', 'handwheelAngle']
COLUMNS_LONGITUDINAL = ['throttle', 'brake', 'vxCG', 'pitchAngle']
COLUMNS_LATERAL = ['handwheelAngle', 'vyCG', 'rollAngle', 'yawAngle']
COLUMNS_MOTION = ['axCG', 'ayCG', 'azCG', 'chassisAccelFL', 'chassisAccelFR', 'chassisAccelRL', 'chassisAccelRR',
                  'deflectionFL', 'deflectionFR', 'horizontalSpeed',  # 'deflectionRL', 'deflectionRR',
                  'pitchAngle', 'pitchRate', 'rollAngle', 'rollRate', 'vxCG', 'vyCG', 'vzCG',
                  'wheelAccelFL', 'wheelAccelFR', 'wheelAccelRL', 'wheelAccelRR', 'yawAngle', 'yawRate']
COLUMN_DERIV_PREFIX = 'deriv_'

COLUMNS = copy.deepcopy(COLUMNS_ALL)
for column in COLUMNS_MOTION:
    new_column = COLUMN_DERIV_PREFIX + column
    COLUMNS.append(new_column)

COLUMNS_MOTION_DERIVS = [COLUMN_DERIV_PREFIX + x for x in COLUMNS_MOTION]
COLUMNS_WITH_GPS_JUMP = ['axCG', 'ayCG', 'vxCG', 'vyCG', 'yawAngle', 'pitchAngle', 'rollAngle']
COLUMNS_DERIV_WITH_GPS_JUMP = [COLUMN_DERIV_PREFIX + x for x in COLUMNS_WITH_GPS_JUMP]

DEFAULT_THRESHOLDS = [
    ('yawAngle', 10),
    ('pitchAngle', 2),
    ('rollAngle', 2),
    ('vxCG', 10),
    ('vyCG', 10)
]
DEFAULT_START = 0
DEFAULT_STOP = None
DEFAULT_STEP = 250
DEFAULT_IMAGE_WIDTH = 1000
DEFAULT_IMAGE_HEIGHT = 500


def get_all_file_paths(data_dir=DATA_DIR):
    logger = logging.getLogger(common.LOG_ROOT)
    file_paths = []

    for dir_path, dir_names, file_names in os.walk(data_dir):
        for file_name in file_names:
            if file_name.endswith('.parquet'):
                file_path = os.path.join(dir_path, file_name)
                file_paths.append(file_path)

    file_paths.sort()
    logger.info('num files: %s', len(file_paths))
    return file_paths


def short_path(file_path):
    return '/'.join(file_path.split('/')[-2:])


def load(file_path):
    # read parquet
    table = pq.read_table(file_path)

    # convert parquet table to pandas dataframe
    df = table.to_pandas()

    return df


def stride_rows(df, stride):
    return df[df.index % stride == 0].reset_index(drop=True)


def add_derivatives(df, strides, columns_to_deriv=COLUMNS_MOTION):
    assert isinstance(strides, list) or isinstance(strides, range)

    logger = logging.getLogger(common.LOG_ROOT)

    derivative_columns = []
    for stride in strides:
        for column in columns_to_deriv:
            new_column = COLUMN_DERIV_PREFIX + '%s_' % stride + column
            derivative_columns.append(new_column)

            df[new_column] = df[column] - df[column].shift(stride)

        # correct for yawAngle sign flip
        deriv_yawAngle = 'deriv_%s_yawAngle' % stride
        indexes = df.index[abs(df[deriv_yawAngle]) > 300]
        for i in indexes:
            df.at[i, deriv_yawAngle] = (180 - abs(df.iloc[i]['yawAngle'])) + (180 - abs(df.iloc[i - stride]['yawAngle']))

        logger.debug('# %s fixed: %s' % (deriv_yawAngle, len(indexes)))

    # replace NaN with zeros in deriv columns
    #values = {COLUMN_DERIV_PREFIX + x: 0 for x in columns_to_deriv}
    #df.fillna(value=values, inplace=True)

    return df, derivative_columns


def clean_discontinuities(df, stride, thresholds=DEFAULT_THRESHOLDS):
    logger = logging.getLogger(common.LOG_ROOT)

    for col_end, threshold in thresholds:
        # find indexes outside of threshold
        column = None
        for _col in df.columns:
            if _col.startswith('deriv') and _col.endswith(col_end):
                column = _col
                break
        if not column:
            logger.warning('could not find %s', col_end)

        indexes = df.index[(df[column] > threshold) | (df[column] < -threshold)]

        columns_deriv_with_gps_jump = []
        for _col_end in COLUMNS_WITH_GPS_JUMP:
            for _col in df.columns:
                if _col.startswith('deriv') and _col.endswith(_col_end):
                    columns_deriv_with_gps_jump.append(_col)
        for i in indexes:
            if numpy.isnan(df.iloc[i - stride]['altitude']):
                logger.debug('GPS NaN at %s : %s -> %s', i,
                        df.iloc[i][columns_deriv_with_gps_jump].to_string(header=False, index=False).replace(os.linesep, ','),
                        df.iloc[i - 1][columns_deriv_with_gps_jump].to_string(header=False, index=False).replace(os.linesep, ','))

                df.at[i, columns_deriv_with_gps_jump] = df.iloc[i - stride][columns_deriv_with_gps_jump]

    return df


def stride_table_rows_and_max_pool_deriv(df, stride):
    # stride original columns
    df_orig = df[df.index % stride == (stride - 1)][COLUMNS_ALL].reset_index(drop=True)

    # max pool derivative columns
    df_deriv = df.groupby(df.index // stride).max()[COLUMNS_MOTION_DERIVS].reset_index(drop=True)

    return pandas.concat([df_orig, df_deriv], axis=1, sort=False)


def display_discontinuities(df, stride, thresholds=DEFAULT_THRESHOLDS):
    logger = logging.getLogger(common.LOG_ROOT)

    for column, index, _df in discontinuity_generator(df):
        logger.debug('%s : %s', column, index)
        display(_df)  # to be used from juypter notebook with "from IPython.display import display"


def discontinuity_generator(df, thresholds=DEFAULT_THRESHOLDS):
    for column, threshold in thresholds:
        indexes = df.index[(df[column] > threshold) | (df[column] < -threshold)]

        for index in indexes:
            yield column, index, df.iloc[index - 1: index + 2, :]


def get_plotly_fig(df, title, columns, start=DEFAULT_START, stop=DEFAULT_STOP, step=DEFAULT_STEP, x_axis=['time']):
    logger = logging.getLogger(common.LOG_ROOT)
    data = []

    if '/' in title:
        title = short_path(title)  # if title is path, reduce to 1 directory and file name

    try:
        x = df[x_axis].values[start:stop:step]
    except TypeError:
        x = list(range(len(df)))

    # replace NaN
    values = {x: 0 for x in columns}
    df = df.fillna(value=values)

    for column in columns:
        if column == 'time' or column == 'distance':
            logger.debug('skipping column: %s', column)
            continue

        y = df[column].values[start:stop:step]

        name = column
        yaxis = 'y'
        if max(abs(y)) > 2:
            yaxis = 'y2'
            name += ' (right axis)'

        trace = plotly.graph_objs.Scatter(
            x=x,
            y=y,
            name=name,
            yaxis=yaxis
        )
        data.append(trace)

    del df

    layout = plotly.graph_objs.Layout(
        title=title,
        yaxis=dict(
            side='left'
        ),
        yaxis2=dict(
            overlaying='y',
            side='right'
        ),
        legend=dict(orientation="h")
    )

    fig = plotly.graph_objs.Figure(data=data, layout=layout)

    return fig


def write_image(fig, image_path, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT):
    logger = logging.getLogger(common.LOG_ROOT)

    # make image directory if necessary
    image_dir = os.path.join('images', image_path.split('/')[-2])
    if not os.path.exists(image_dir):
        logger.info('mkdir: %s', image_dir)
        os.mkdir(image_dir)

    plotly.io.write_image(fig, image_path, width=width, height=height)
    logger.info('wrote: %s', image_path)
    
    return fig


def plot(df, file_path, columns, title='', plot=True, write=False, start=DEFAULT_START, stop=DEFAULT_STOP, step=DEFAULT_STEP, x_axis='time'):
    if start == 'middle':
        start = len(df) // 2
    if stop == 'middle':
        stop = start + 100
    image_path = None
    if title:
        title += '<br>'
    title += 'start {start}, stop {stop}, step {step}'.format(start=start, stop='end' if stop is None else stop, step=step)
    if write:
        image_dir = os.path.join('images', file_path.split('/')[-2])
        columns_hash = hashlib.md5()
        columns_hash.update(" ".join(columns).encode('utf-8'))
        image_path = os.path.join(image_dir, os.path.splitext(file_path.split('/')[-1])[0] +
                                  '-%s-%s-%s-%s' % (start, stop, step, columns_hash.hexdigest()[:7]) + '.jpeg')
        title = image_path + '<br>' + title

    fig = get_plotly_fig(df, title, columns=columns, start=start, stop=stop, step=step, x_axis=x_axis)

    if plot:
        plotly.offline.iplot(fig)
    if write:
        fig = write_image(fig=fig, image_path=image_path)

    return fig, image_path


def get_data_sets(df, strides=range(1, 11), train_percent=0.9, dev_percent=0.05, test_percent=0.05):
    assert (train_percent + dev_percent + test_percent) == 1

    # build data (input) DataFrame
    df_data = copy.deepcopy(df)
    df_data, deriv_columns = add_derivatives(df_data, strides=strides, columns_to_deriv=df_data.columns)
    data_columns = list(df_data.columns)

    # build labels
    df_labels = copy.deepcopy(df[COLUMNS_MOTION])
    df_labels, deriv_columns = add_derivatives(df_labels, strides=[1], columns_to_deriv=COLUMNS_MOTION)
    df_labels = df_labels[deriv_columns]

    # update label column names to prepend "label_"
    label_map = {}
    for col in df_labels.columns:
        label_map[col] = 'label_' + col
    df_labels.rename(index=str, columns=label_map, inplace=True)
    label_columns = df_labels.columns

    # associate input data (prior timestamp) & output labels (next timestamp) on same row
    df_data = df_data.iloc[:-1].reset_index(drop=True)
    df_labels = df_labels.iloc[1:].reset_index(drop=True)

    # concatenate data to labels
    df_out = pandas.concat([df_data, df_labels], axis=1, sort=False)
    del df_data
    del df_labels

    df_out.dropna(axis=0, inplace=True)

    #df_out = df_out.iloc[len(strides):].reset_index(drop=True)

    # shuffle order of combined data frame
    df_out = df_out.sample(frac=1)

    # set indexes to split across 3 sets
    train_rows = (0, round(len(df) * train_percent))
    dev_rows = (train_rows[1], train_rows[1] + round(len(df) * dev_percent))
    test_rows = (dev_rows[1], dev_rows[1] + round(len(df) * test_percent))

    # split into 3 sets
    df_train = df_out.iloc[train_rows[0]: train_rows[1]]
    df_dev = df_out.iloc[dev_rows[0]: dev_rows[1]]
    df_test = df_out.iloc[test_rows[0]: test_rows[1]]

    return df_train, df_dev, df_test, data_columns, label_columns

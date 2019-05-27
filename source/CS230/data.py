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

DATA_DIR = os.path.join(os.path.abspath(__file__).split('CS230_project')[0], 'CS230_project', 'data')

COLUMNS_ORIG = ['time', 'handwheelAngle', 'throttle', 'brake', 'latitude', 'longitude', 'altitude', 'horizontalSpeed', 'vxCG', 'vyCG', 'yawAngle', 'pitchAngle', 'rollAngle', 'distance']
COLUMNS_TO_DERIV = ['yawAngle', 'pitchAngle', 'rollAngle', 'horizontalSpeed', 'distance', 'vxCG', 'vyCG']
COLUMN_DERIV_PREFIX = 'deriv_'

COLUMNS = copy.deepcopy(COLUMNS_ORIG)
for column in COLUMNS_TO_DERIV:
    new_column = COLUMN_DERIV_PREFIX + column
    COLUMNS.append(new_column)

DERIV_COLUMNS = [COLUMN_DERIV_PREFIX + x for x in COLUMNS_TO_DERIV]
COLUMNS_WITH_GPS_JUMP = ['horizontalSpeed', 'vxCG', 'vyCG', 'yawAngle', 'pitchAngle', 'rollAngle']
DERIV_COLUMNS_WITH_GPS_JUMP = [COLUMN_DERIV_PREFIX + x for x in COLUMNS_WITH_GPS_JUMP]

DEFAULT_THRESHOLDS = [
    ('deriv_yawAngle', 10),
    ('deriv_pitchAngle', 2),
    ('deriv_rollAngle', 2),
    ('deriv_distance', 10),
    ('deriv_vxCG', 10),
    ('deriv_vyCG', 10)
]
DEFAULT_START = 0
DEFAULT_STOP = None
DEFAULT_STEP = 250
DEFAULT_IMAGE_WIDTH = 1000
DEFAULT_IMAGE_HEIGHT = 500


def get_all_file_paths(data_dir=DATA_DIR, exclude=['grandsport.parquet', '250lm.parquet']):
    logger = logging.getLogger(common.LOG_ROOT)
    file_paths = []

    for dir_path, dir_names, file_names in os.walk(data_dir):
        for file_name in file_names:
            if file_name.endswith('.parquet') and file_name not in exclude:
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
    return df[df.index % stride == (stride - 1)].reset_index(drop=True)


def add_derivatives(df, stride, columns_to_deriv=COLUMNS_TO_DERIV):
    logger = logging.getLogger(common.LOG_ROOT)

    for column in columns_to_deriv:
        new_column = COLUMN_DERIV_PREFIX + column

        df[new_column] = df[column] - df[column].shift(stride)

    # correct for yawAngle sign flip
    indexes = df.index[abs(df['deriv_yawAngle']) > 300]
    for i in indexes:
        df.at[i, 'deriv_yawAngle'] = (180 - abs(df.iloc[i]['yawAngle'])) + (180 - abs(df.iloc[i - stride]['yawAngle']))

    logger.debug('# deriv_yawAngle fixed: %s' % len(indexes))

    # replace NaN with zeros in deriv columns
    values = {COLUMN_DERIV_PREFIX + x: 0 for x in columns_to_deriv}
    df.fillna(value=values, inplace=True)

    return df


def clean_discontinuities(df, stride, thresholds=DEFAULT_THRESHOLDS):
    logger = logging.getLogger(common.LOG_ROOT)

    for column, threshold in thresholds:
        # find indexes outside of threshold
        indexes = df.index[(df[column] > threshold) | (df[column] < -threshold)]

        for i in indexes:
            if numpy.isnan(df.iloc[i - stride]['altitude']):
                logger.debug('GPS NaN at %s : %s -> %s', i,
                        df.iloc[i][DERIV_COLUMNS_WITH_GPS_JUMP].to_string(header=False, index=False).replace(os.linesep, ','),
                        df.iloc[i - 1][DERIV_COLUMNS_WITH_GPS_JUMP].to_string(header=False, index=False).replace(os.linesep, ','))

                df.at[i, DERIV_COLUMNS_WITH_GPS_JUMP] = df.iloc[i - stride][DERIV_COLUMNS_WITH_GPS_JUMP]

    return df


def stride_table_rows_and_max_pool_deriv(df, stride):
    # stride original columns
    df_orig = df[df.index % stride == (stride - 1)][COLUMNS_ORIG].reset_index(drop=True)

    # max pool derivative columns
    df_deriv = df.groupby(df.index // stride).max()[DERIV_COLUMNS].reset_index(drop=True)

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


def get_plotly_fig(df, title, columns, start=DEFAULT_START, stop=DEFAULT_STOP, step=DEFAULT_STEP):
    logger = logging.getLogger(common.LOG_ROOT)
    data = []

    if '/' in title:
        title = short_path(title)  # if title is path, reduce to 1 directory and file name

    x = df['time'].values[start:stop:step]

    # replace NaN
    values = {x: 0 for x in columns}
    df = df.fillna(value=values)

    for column in columns:
        if column == 'time' or column == 'distance':
            logger.debug('skipping column: %s', column)
            continue

        y = df[column].values[start:stop:step]

        if max(abs(y)) > 1:
            yaxis = 'y2'
            name = column + ' (right axis)'
        else:
            yaxis = 'y'
            name = column

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
        )
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


def plot_source(df, file_path, plot=True, write=False):
    start = DEFAULT_START
    stop = DEFAULT_STOP
    step = DEFAULT_STEP
    columns = COLUMNS_ORIG
    image_path = None
    title = '<b>source data zoomed-out</b>: sampled with 1D stride={step}'.format(step=step)
    if write:
        image_dir = os.path.join('images', file_path.split('/')[-2])
        image_path = os.path.join(image_dir, os.path.splitext(file_path.split('/')[-1])[0] + '-%s-%s-%s' % (start, stop, step) + '.jpeg')
        title = image_path + '<br>' + title

    fig = get_plotly_fig(df, title, columns=columns, start=start, stop=stop, step=step)

    if plot:
        plotly.offline.iplot(fig)
    if write:
        fig = write_image(fig=fig, image_path=image_path)

    return fig, image_path


def plot_source_zoomed(df, file_path, plot=True, write=False, start=None, stop=None, step=None):
    if start is None:
        start = len(df) // 2
    if stop is None:
        stop = start + 200
    if step is None:
        step = 1
    columns = COLUMNS_ORIG
    image_path = None
    title = '<b>source data zoomed-in</b>: indexes {start} : {stop}'.format(start=start, stop=stop)
    if write:
        image_dir = os.path.join('images', file_path.split('/')[-2])
        image_path = os.path.join(image_dir, os.path.splitext(file_path.split('/')[-1])[0] + '-%s-%s-%s' % (start, stop, step) + '_zoom.jpeg')
        title = image_path + '<br>' + title

    fig = get_plotly_fig(df, title, columns=columns, start=start, stop=stop, step=step)

    if plot:
        plotly.offline.iplot(fig)
    if write:
        fig = write_image(fig=fig, image_path=image_path)

    return fig, image_path


def plot_derivatives(df, file_path, plot=True, write=False):
    start = DEFAULT_START
    stop = DEFAULT_STOP
    step = DEFAULT_STEP
    columns = DERIV_COLUMNS
    image_path = None
    title = '<b>derivatives zoomed-out</b>: sampled with 1D stride={step}'.format(step=step)
    if write:
        image_dir = os.path.join('images', file_path.split('/')[-2])
        image_path = os.path.join(image_dir, os.path.splitext(file_path.split('/')[-1])[0] + '-%s-%s-%s' % (start, stop, step) + '_deriv.jpeg')
        title = image_path + '<br>' + title

    fig = get_plotly_fig(df, title, columns=columns, start=start, stop=stop, step=step)

    if plot:
        plotly.offline.iplot(fig)
    if write:
        fig = write_image(fig=fig, image_path=image_path)

    return fig, image_path


def plot_derivatives_zoomed(df, file_path, plot=True, write=False, start=None, stop=None, step=None):
    if start is None:
        start = len(df) // 2
    if stop is None:
        stop = start + 200
    if step is None:
        step = 1
    columns = DERIV_COLUMNS
    image_path = None
    title = '<b>derivatives zoomed-in</b>: indexes {start} : {stop}'.format(start=start, stop=stop)
    if write:
        image_dir = os.path.join('images', file_path.split('/')[-2])
        image_path = os.path.join(image_dir, os.path.splitext(file_path.split('/')[-1])[0] + '-%s-%s-%s' % (start, stop, step) + '_deriv_zoom.jpeg')
        title = image_path + '<br>' + title

    fig = get_plotly_fig(df, title, columns=columns, start=start, stop=stop, step=step)

    if plot:
        plotly.offline.iplot(fig)
    if write:
        fig = write_image(fig=fig, image_path=image_path)

    return fig, image_path

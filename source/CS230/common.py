import logging


LOG_ROOT = 'CS230'


def get_logger():
    logger = logging.getLogger(LOG_ROOT)
    logger.setLevel(logging.DEBUG)

    # create a file handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H-%M-%S')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger

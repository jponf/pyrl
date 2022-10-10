# -*- coding: utf-8 -*-

"""Utility module to ease the access to a common logger accros the different
modules of the project.
"""


import collections
import copy
import logging
import sys

from pyrl.util.type_utils import StrEnum

###############################################################################


class LoggingLevelName(StrEnum):
    """Logging message levels names."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


LEVELS = collections.OrderedDict(
    [
        (LoggingLevelName.NOTSET, logging.NOTSET),
        (LoggingLevelName.DEBUG, logging.DEBUG),
        (LoggingLevelName.INFO, logging.INFO),
        (LoggingLevelName.WARNING, logging.WARNING),
        (LoggingLevelName.ERROR, logging.ERROR),
        (LoggingLevelName.CRITICAL, logging.CRITICAL),
    ],
)

LOGGER_NAME = "PyRL"


###############################################################################


def get_logger():
    """Access the pydgga package logger.

    The logger is not initialized and may not print anything. To ensure
    its proper initialization, please call :func:`pyrl.util.logging.setup`.

    :return: The pydgga package logger instance.
    :rtype: logging.Logger.
    """
    return logging.getLogger(LOGGER_NAME)


def set_up_logger(level):
    """Sets up the robotrl package logger.

    This function should only be called once, doing so many times may
    result in undefined behaviour such as, repeated output or unexpected
    crashes..

    :param Union[int, str] level: The minimum message level that will be
        outputted by the logger. If it is a string, it must match the names
        of the levels defined in the standard Python's logging module.
    """
    date_fmt = "%d-%m-%Y %H:%M:%S"
    msg_fmt = "[%(asctime)s - " + LOGGER_NAME + " - %(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt=msg_fmt, datefmt=date_fmt)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)

    if isinstance(level, str):
        level = LEVELS[level]

    logger = get_logger()
    for handler in copy.copy(logger.handlers):  # .copy():
        logger.removeHandler(handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(level)

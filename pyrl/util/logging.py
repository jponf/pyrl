# -*- coding: utf-8 -*-

"""Utility module to ease the access to a common logger accros the different
modules of the project.
"""

from __future__ import (
    absolute_import, print_function, division, unicode_literals
)

import collections
import copy
import logging
import sys


###############################################################################

LEVELS = collections.OrderedDict([("NOTSET", 0), ("DEBUG", 10),
                                  ("INFO", 20), ("WARNING", 30),
                                  ("ERROR", 40), ("CRITICAL", 50)])


###############################################################################

def get_logger():
    """Access the pydgga package logger.

    The logger is not initialized and may not print anything. To ensure
    its proper initialization, please call :func:`pyrl.util.logging.setup`.

    :return: The pydgga package logger instance.
    :rtype: logging.Logger.
    """
    return logging.getLogger("RobotRL")


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
    msg_fmt = '[%(asctime)s - %(levelname)s] %(message)s'
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

# -*- coding: utf-8 -*-

"""Module dedicated to operating system convenience functions and other
utilities, such as checking the operating system type or modifying signal
handlers. It also contains functions that mimic the behaviour of common
shell commands and specific constants.
"""

import os
import os.path
import platform
import stat


# Module constants
###############################################################################

OS_WINDOWS_NAMES = {'nt'}  # Cygwin reports posix (check)
OS_LINUX_NAMES = {'posix'}
OS_MAC_NAMES = {'posix'}

OS_POSIX_NAMES = OS_LINUX_NAMES | OS_MAC_NAMES

PLATFORM_WINDOWS_SYS = {'Windows'}
PLATFORM_LINUX_SYS = {'Linux'}
PLATFORM_MAC_SYS = {'Darwin'}


# O.S test methods
###############################################################################

def is_windows():
    """Convenience function that tests different information sources to verify
    whether the operating system is Windows.

    :return: True if the operating system is windows, False otherwise.
    :rtype: bool
    """
    return (os.name in OS_WINDOWS_NAMES and
            platform.system() in PLATFORM_WINDOWS_SYS)


def is_linux():
    """Convenience function that tests different information sources to verify
    whether the operating system is a Linux based O.S.

    :return: True if the operating system is Linux, False otherwise.
    :rtype: bool
    """
    return (os.name in OS_LINUX_NAMES and
            platform.system() in PLATFORM_LINUX_SYS)


def is_mac():
    """Convenience function that tests different information sources to verify
    whether the operating system is MacOS/OS X.

    :return: True if the operating system is MacOS, False otherwise.
    :rtype: bool
    """
    return (os.name in OS_MAC_NAMES and
            platform.system() in PLATFORM_MAC_SYS)


def is_posix():
    """Convenience function that tests different information sources to verify
    whether the operating system is POSIX compliant.

    .. note::
        No assumption is made reading the POSIX level compliance.

    :return: True if the operating system is MacOS, False otherwise.
    :rtype: bool
    """
    return os.name in OS_POSIX_NAMES


# File utilities
###############################################################################

def is_executable(path):
    """Tests if the specified path corresponds to an executable file. This is,
    it is a file and also has the appropriate executable bit set.

    :param path: Path to the file to test.
    :return: True if the file exists and is executable, False otherwise.
    :rtype: bool
    """
    if os.path.isfile(path):
        f_stat = os.stat(path)
        is_exe = f_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return bool(is_exe)
    return False


def which(program, test_common_extensions=True):
    """Returns  the  path  of the file (or link) which would be executed in the
    current environment, had its argument been given as a command in a strictly
    POSIX-conformant shell. It does this by searching the PATH for executable
    files  matching the name of the argument.

    :param str program: Path or name of the program to find.
    :param bool test_common_extensions: Append extensions that are common on
        the operating system when looking for the program.
    :return: The path to the specified `program` or empty string if it could
        not be found.
    :rtype: str
    """
    f_path, _ = os.path.split(program)
    if f_path:
        if is_executable(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_executable(exe_file):
                return exe_file
            elif test_common_extensions:
                if is_windows() and is_executable(exe_file + ".exe"):
                    return exe_file + ".exe"

    return ""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, print_function, division, unicode_literals
)
from builtins import *

from setuptools import find_packages, setup


# Comma separated list of names and emails
authors = "Josep Pon"

emails = "Josep.Pon@gft.com"

# Short description
description = ""

# Long description
with open("README.md", encoding="utf-8") as f:
    readme = f.read()

# Requirements
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [x for x in map(str.strip, f.read().splitlines())
                    if x and not x.startswith("#")]


# Additional keyword arguments
kwargs = {
    "entry_points": {
        "console_scripts": []
    },
}


################################################################################

setup(
    name='robotrl',
    version="0.0.1",
    description=description,
    long_description=readme,
    author=authors,
    author_email=emails,
    url="https://git.gft.com/ai-practice-es/robotic-arm-rl",
    license="",
    keywords='ros open_manipulator ppo reinforcement learning',
    install_requires=requirements,
    packages=find_packages(),
    package_data={},
    platforms='any',
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    **kwargs
)
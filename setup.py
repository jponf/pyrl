# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, print_function, division, unicode_literals
)

import codecs
import os.path

from setuptools import find_packages, setup


###############################################################################

def read_rel(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read_rel(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


###############################################################################

# Comma separated list of names and emails
AUTHORS = "Josep Pon, Guillem Orellana"

EMAILS = "jponfarreny@gmail.com, guillem.orellana@gmail.com"

# Short description
DESCRIPTION = ""

# Long description
with open("README.md", "rt") as f:
    README = f.read()

# Requirements
with open("requirements.txt", "rt") as f:
    REQUIREMENTS = [x for x in map(str.strip, f.read().splitlines())
                    if x and not x.startswith("#")]

KEYWORDS = ["Reinforcement Learning", "DDPG", "TD3", "SAC",
            "Hindsight Experience Replay", "HER"]


# Additional keyword arguments
kwargs = {
    "entry_points": {
        "console_scripts": ["pyrl=pyrl.__main__:run_main"]
    },
}


###############################################################################

setup(
    name='pyrl',
    version=get_version(os.path.join("pyrl", "__init__.py")),
    description=DESCRIPTION,
    long_description=README,
    author=AUTHORS,
    author_email=EMAILS,
    url="https://git.gft.com/ai-practice-es/pyrl",
    license="",
    keywords=KEYWORDS,
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    package_data={},
    platforms='any',
    zip_safe=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    **kwargs
)

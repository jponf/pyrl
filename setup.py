# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, print_function, division, unicode_literals
)

from setuptools import find_packages, setup


# Comma separated list of names and emails
AUTHORS = "Josep Pon, Guillem Orellana"

EMAILS = "Josep.Pon@gft.com, Guillem.Orellana@gft.com"

# Short description
DESCRIPTION = ""

# Long description
with open("README.md", encoding="utf-8") as f:
    README = f.read()

# Requirements
with open("requirements.txt", encoding="utf-8") as f:
    REQUIREMENTS = [x for x in map(str.strip, f.read().splitlines())
                    if x and not x.startswith("#")]

KEYWORDS = ["Reinforcement Learning", "DDPG", "TD3",
            "Hindsight Experience Replay", "HER"]


# Additional keyword arguments
kwargs = {
    "entry_points": {
        "console_scripts": []
    },
}


###############################################################################

setup(
    name='pyrl',
    version="0.0.1",
    description=DESCRIPTION,
    long_description=README,
    author=AUTHORS,
    author_email=EMAILS,
    url="https://git.gft.com/ai-practice-es/robotic-arm-rl",
    license="",
    keywords=KEYWORDS,
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    package_data={},
    platforms='any',
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
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

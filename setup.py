#!/usr/bin/env python

import os
from setuptools import find_packages, setup
import numpy as np


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="psat",
    version="0.0.1",
    author="Jamie Morton",
    author_email="jamietmorton@gmail.com",
    description=("Permutative statistical analysis toolbox"),
    packages=find_packages(),
    long_description=read('README.md'),
    license='MIT',
    include_dirs=[np.get_include()],
    install_requires=[
        'pandas',
        'numpy >= 1.7',
        'scipy >= 0.14.0',
        ],
    extras_require={'test': ["nose >= 0.10.1", "pep8", "flake8",
                             "python-dateutil"]},
)

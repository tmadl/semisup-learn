#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    "sklearn",
    "scipy",
    "numpy",
    "matplotlib"
]

test_requirements = []

setup(
    name='semisup_learn',
    version='0.0.1',
    description="Semisupervised Learning Framework",
    url='https://github.com/tmadl/semisup-learn',
    packages=[
        'methods', 'frameworks'
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='semisup-learn',
)
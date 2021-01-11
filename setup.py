"""
This file contains the necessary setup steps in order to launch Dataflow with
the necessary packages and software installed.
"""
#!/usr/bin/python

import setuptools
from setuptools import find_packages

REQUIRED = [
    "apache-beam[gcp]==2.25.0",
    "tensorflow==2.3.1",
    "pandas==1.1.4",
    "Pillow==8.0.1",
    "gcsfs==0.7.1"
]


setuptools.setup(
    name="tfrecords-centernet",
    description="Dataflow Pipeline that generated tfrecords",
    version='1.0.0',
    author='Stef Ruinard',
    author_email='sr.ruinard@gmail.com',
    install_requires=[REQUIRED],
    packages=find_packages(
        exclude=["trainer", "core"]),
    py_modules=['config'],
    scripts=[],
)

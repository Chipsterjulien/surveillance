#! /usr/bin/env python3
# -*- encoding: utf-8 -*-


# see :
# https://github.com/pypa/sampleproject/blob/master/setup.py
# https://github.com/kennethreitz/setup.py

# https://packaging.python.org/tutorials/packaging-projects/
# https://wiki.labomedia.org/index.php/Cr%C3%A9er_son_propre_package_python
# https://medium.com/38th-street-studios/creating-your-first-python-package-181c5e31f3f8

# Test unitaire:
# https://docs.python.org/3.7/library/test.html#writing-unit-tests-for-the-test-package


# from setuptools import setup, find_packages
from distutils.core import setup, find_packages


setup(
    name='surveillance',
    version='0.0.1',
    author='Freyermuth Julien',
    author_email="",
    description="A basic motion's detector",
    url='',
    download_url='',
    license='BSD',
    keywords=["motion", "opencv"],
    classifiers=[],
    py_modules=[],
    DATA_FILES=[('/etc/surveillance', ['cfg/surveillance_sample.conf'])],
    SCRIPTS=['surveillance'],
    include_package_data=True,
    requires=['imutils', 'opencv-python'],
    packages=find_packages(),
)

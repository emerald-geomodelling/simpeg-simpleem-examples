#!/usr/bin/env python

import setuptools
import os

setuptools.setup(
    name='simpegsimpleem',
    version='0.0.1',
    description='',
    long_description="",
    long_description_content_type="text/markdown",
    author='Egil Moeller ',
    author_email='em@emeraldgeo.no',
    url='https://github.com/emerald-geomodelling/experimental-simpeg-ext',
    packages=setuptools.find_packages(),
    install_requires=[
        "cython",
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "discretize",
        "simpeg @ git+https://github.com/emerald-geomodelling/simpeg.git@em1d_updates#egg=SimPEG",
        "simpegEM1D",
        "libaarhusxyz @ git+https://github.com/emerald-geomodelling/libaarhusxyz.git@normalization#egg=libaarhusxyz",
    ],
)

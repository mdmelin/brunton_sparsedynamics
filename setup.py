#!/usr/bin/env python
# Install script for some random utils.
# Max Melin

import os
from setuptools import setup

setup(
    name = 'sparse_dynamics',
    version = '0.1.0',
    author = 'Max Melin',
    author_email = 'mmelin@g.ucla.edu',
    description = 'Helper functions inferring sparse dynamics. Python port of Brunton et al. 2016.',
    packages = ['sparsedynamics']
)
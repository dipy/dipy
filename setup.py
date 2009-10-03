#!/usr/bin/env python
''' Installation script for dipy package '''

from glob import glob
from distutils.core import setup

setup(name='dipy',
      version='0.1a',
      description='Diffusion utilities in Python',
      author='DIPY ython team',
      author_email='matthew.brett@gmail.com',
      url='http://github.com/matthew-brett/dipy',
      packages=['dipy', 'dipy.io', 'dipy.viz'],
      scripts=glob('scripts/*.py')
      )


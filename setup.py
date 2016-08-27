# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 01:47:02 2016

@author: sramena1
"""

from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='gmodel',
      version='0.1',
      description="""A computational model for contour grouping, object proposal
      and border ownership""",
      url='https://github.com/sramena1/gmodel',  
      author='Sudarshan Ramenahalli, Johns Hopkins University',
      author_email='sudarshan.rg@gmail.com',
      license='MIT',
      packages=['gmodel','stimuli'],
      zip_safe=False)
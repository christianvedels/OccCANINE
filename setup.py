""" Setup
"""


from os import path

from codecs import open

from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


exec(open('hisco/version.py').read())
setup(
    name='hisco',
    version=__version__,
    description='Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE.',
    author='Christian Vedel',
    author_email='christian-vs@sam.sdu.dk',
    url='https://github.com/christianvedels/OccCANINE',
    packages=find_packages(exclude=['cfgs', 'docs', 'examples', 'helpers', 'tests']),
)
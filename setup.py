import setuptools
from codecs import open  # To use a consistent encoding
import os
import sys


__name__ = 'CS230'
__version__ = '0.0.1'
__author__ = 'David Casterton'
__email__ = 'david.casterton@gmail.com'


INSTALL_REQUIRES = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        requirements = f.readlines()
    INSTALL_REQUIRES.extend(requirements)

THISDIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(THISDIR, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DATA_FILES = []
PACKAGE_DATA = {
    '': ['']
}

setuptools.setup(
    name=__name__,  # Required
    version=__version__,  # Required
    description='',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    author=__author__,  # Optional
    author_email=__email__,  # Optional
    package_dir={'': 'source'},
    packages=setuptools.find_packages('source'),
    install_requires=INSTALL_REQUIRES,
    package_data=PACKAGE_DATA,  # Optional
    data_files=DATA_FILES,
)
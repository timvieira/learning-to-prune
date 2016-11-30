import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

# Changeprop seems to run slower under clang.
#os.environ["CC"] = "clang++"
#os.environ["CXX"] = "clang++"


setup(name='ldp',
      version='1.0',
      description='',
      author='Tim Vieira',
      packages=['ldp'],
      install_requires=[
          'path.py',
          'psutil==2.1',
          'murmurhash==0.24',
          'nltk>=3.0',
      ],
      include_dirs=[np.get_include(),
                    '/cm/local/apps/boost/1.58.0/include/',     # MARCC grid.
                    os.path.expanduser('~/anaconda/include/')],
      library_dirs = ['/usr/local/lib'],
      ext_modules = cythonize(['ldp/**/*.pyx']))

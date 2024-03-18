from skbuild import setup
import numpy

setup(
    name='encodecpp',
    packages=['encodecpp'],
    include_dirs=[numpy.get_include()]
)

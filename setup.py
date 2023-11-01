import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("marriage_code.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    compiler_directives={'boundscheck': False}
)

setup(
    ext_modules=cythonize("graph_current_distributions.pyx"),
    include_dirs=[numpy.get_include()]
)
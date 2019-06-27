

from distutils.core import setup, Extension
import os
import sysconfig
import numpy

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra", '-L/usr/local/lib/','-lgslcblas', '-lgsl' ,'-lm']

module1 = Extension('cfar',
                    include_dirs = [
                     #       '/media/nasir/Drive/code/ibis_extension/ibis_lib',
                           '/media/nasir/Drive1/code/SAR/AutomatedSARShipDetection/python_cfar/','/usr/local/lib/', numpy.get_include()],
                    library_dirs = ['/usr/local/lib/'],
                    libraries = ['opencv_core', 'opencv_highgui','opencv_video','opencv_videoio', 'gsl', 'gslcblas'],
                    extra_compile_args=extra_compile_args,
                    sources = [ 'main.cpp']
                     )


setup (name = 'cfar',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])


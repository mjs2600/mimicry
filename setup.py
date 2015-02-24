from distutils.core import setup

setup(
    name='mimicry',
    version='0.0.1',
    author='M. Simpson, J. McGehee',
    author_email= 'mjs2600@gmail.com, jlmcgehee21@gmail.com',
    packages=['mimicry'],
#     scripts=['bin/example1.py','bin/example2.py'], #Any scripts we may want
#     url='http://pypi.python.org/pypi/mimicry/', #If we actually want to upload this to PyPI
    license='MIT', #Unless you prefer another
    description='MIMIC Randomized Optimization Algorithm in Python',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.8, <2.0",
    ],
)
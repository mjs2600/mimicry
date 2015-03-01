from setuptools import setup

setup(
    name='mimicry',
    version='0.2.2',
    author='M. Simpson, J. McGehee',
    author_email= 'mjs2600@gmail.com, jlmcgehee21@gmail.com',
    packages=['mimicry'],
#     scripts=['bin/example1.py','bin/example2.py'], #Any scripts we may want
#     url='http://pypi.python.org/pypi/mimicry/', #If we actually want to upload this to PyPI
    license='MIT', #Unless you prefer another
    description='MIMIC Randomized Optimization Algorithm in Python',
    long_description='MIMIC Randomized Optimization Algorithm in Python',
    install_requires=[
        "numpy >= 1.8, <2.0",
        "networkx >= 1.9.1, <2.0",
        "scikit-learn >= 0.15.2, <1.0",
        "matplotlib >= 1.3.0, <1.5",
    ],
)

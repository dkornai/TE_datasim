from setuptools import setup, find_packages

setup(
    name='te_datasim',
    version='0.1',
    packages=find_packages(),
    description='Simulate time series data for benchmarking transfer entropy estimation methods',
    author='Daniel Kornai',
    install_requires=[
        'numpy>=1.20',
    ]
)
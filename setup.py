
from setuptools import setup, find_packages

setup(
    name='landmark',
    install_requires=[
        'lime', 'seaborn'
    ], 
    packages=find_packages(exclude=('test', 'data', 'evaluation', 'wrapper'))
)

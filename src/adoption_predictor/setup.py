from setuptools import setup, find_packages
import re
import ast

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('adoption_predictor/__init__.py', 'rb') as f:
    VERSION = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
    name='adoption_predictor',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Dependencies
    ],
)

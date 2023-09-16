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
    # author='Your Name',
    # author_email='your@email.com',
    # description='A simple Python module with a single class',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='https://github.com/yourusername/my_module',
    # classifiers=[
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.7',
    #     'Programming Language :: Python :: 3.8',
    #     'Programming Language :: Python :: 3.9',
    # ],
)

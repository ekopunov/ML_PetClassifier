from setuptools import setup, find_packages
import re
import ast

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('gcloud_downloader/__init__.py', 'rb') as f:
    VERSION = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
    name='gcloud_downloader',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "google-cloud-core==2.3.3"
    ],
)

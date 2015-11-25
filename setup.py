#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


ver_file = os.path.join('latte', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# with open('README.rst') as readme_file:
#     readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read().replace('.. :changelog:', '')

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)

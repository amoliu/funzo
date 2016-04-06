import io
import os
import re
import sys
import subprocess


def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         'funzo'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('funzo')

    return config

DESCRIPTION = "funzo: python toolkit for inverse reinforcement learning"
LONG_DESCRIPTION = """
funzo: python toolkit for inverse reinforcement learning
==========================================================
This is a toolkit for inverse reinforcement learning and apprenticeship
learning using python. It includes a number of reinforcement learning domains
and planners. It further contains implementations of various IRL algorithms.
The implementations strive for clean interfaces to allow ease of use and
extension with new algorithms.
For more information, visit https://github.com/makokal/funzo
"""
NAME = "funzo"
AUTHOR = "Billy Okal"
AUTHOR_EMAIL = "sudo@makokal.com"
URL = 'https://github.com/makokal/funzo'
DOWNLOAD_URL = 'https://github.com/makokal/funzo'
LICENSE = 'MIT'
VERSION = version('funzo/__init__.py')


def setup_package():
    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython()

    try:
        setup(name='funzo',
              author=AUTHOR,
              author_email=AUTHOR_EMAIL,
              url=URL,
              download_url=DOWNLOAD_URL,
              description=DESCRIPTION,
              long_description=LONG_DESCRIPTION,
              version=VERSION,
              license=LICENSE,
              configuration=configuration,
              classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Natural Language :: English',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.5'])
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return


if __name__ == '__main__':
    setup_package()

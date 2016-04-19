

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('funzo', parent_package, top_path)

    config.add_subpackage('domains')
    config.add_subpackage('irl')
    config.add_subpackage('irl/tests')
    config.add_subpackage('models')
    config.add_subpackage('models/tests')
    config.add_subpackage('planners')
    config.add_subpackage('planners/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')
    config.add_subpackage('representation')
    config.add_subpackage('representation/tests')
    config.add_data_files('data/DUMMY.txt')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

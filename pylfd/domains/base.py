

from abc import ABCMeta
from abc import abstractmethod


class Domain(object):
    """ Domain interface

    domain summarizes the following:

    MDP states and actions
    MDP dynamics/transitions

    """

    __meta__ = ABCMeta

    def __init__(self, kind='discrete'):
        super(Domain, self).__init__()
        self.kind = kind

    @abstractmethod
    def visualize(self):
        pass



from abc import ABCMeta
from abc import abstractmethod


class Domain(object):
    """ Domain interface

    Domains are extensions of MDPs, have all mdp relevant information

    domain summarizes the following:

    MDP states and actions
    MDP dynamics/transitions

    MDP contains: discounting, terminal states?

    """

    __meta__ = ABCMeta

    def __init__(self, kind='discrete'):
        super(Domain, self).__init__()
        self.kind = kind

    @abstractmethod
    def visualize(self):
        pass

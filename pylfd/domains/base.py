

from abc import ABCMeta
from abc import abstractmethod

import six


class Domain(six.with_metaclass(ABCMeta)):

    """ Domain interface

    Domains are extensions of MDPs, have all mdp relevant information

    domain summarizes the following:

    MDP states and actions
    MDP dynamics/transitions

    MDP contains: discounting, terminal states?

    """

    def __init__(self, kind='discrete'):
        self.kind = kind

    @abstractmethod
    def visualize(self, ax, **kwargs):
        """ visualize a domain

        Parameters
        -----------
        ax : `matplotlib` axes
            Axes on which to visualize the domain
        kwargs : dict
            Optional key-world arguiments

        Returns
        --------
        ax : `matplotlib` axes
            The axis with the visual elements on the domain drawn

        """
        return NotImplementedError('This method is abstract')

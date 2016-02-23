"""
Base interfaces for domains (worlds) which contain all relavant information
about the states and actions in MDPs used by the different functions (reward
controller, transition)

"""

import six

from abc import ABCMeta
from abc import abstractmethod


from ..base import Model


class Domain(six.with_metaclass(ABCMeta, Model)):

    """ Domain (World) interface

    Domains expand on the abstraction in the MDP, they represent the concrete
    details of the task. i.e. the environment.

    """

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (absorbing state) """
        raise NotImplementedError('Abstract method')

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
        raise NotImplementedError('This method is abstract')

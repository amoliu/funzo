"""
Base interfaces for domains (worlds) representing MDPs

"""

import six

from abc import ABCMeta
from abc import abstractmethod


from ..base import Model


class Domain(six.with_metaclass(ABCMeta, Model)):

    """ Domain interface

    Domains expand on the abstraction in the MDP, they represent the concrete
    details of the task. i.e. the environment.

    """

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

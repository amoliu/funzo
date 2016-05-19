"""
Controllers (transition functions) for social navigation

"""

from __future__ import division

import numpy as np

from ..base import model_domain
from ...models import MDPLocalControler

from .nav_world import SocialNavigationWorld


class LinearController(MDPLocalControler):
    """ Linear local controller """
    def __init__(self, domain=None):
        super(LinearController, self).__init__(domain)
        self._domain = model_domain(domain, SocialNavigationWorld)

    def __call__(self, state, action, duation, **kwargs):
        """ Evaluate transition function

        Returns
        --------
        n_s : array-like
            A list of all possible next states [(prob, state)]

        """
        vmax = kwargs.get('max_speed', 1.0)

        state_ = self._domain.states[state]
        action_ = self._domain.actions[action]

        nx = state_[0] + np.cos(action) * duration
        ny = state_[1] + np.sin(action) * duration

        # noise = np.random.normal(0.0, scale=0.01)
        # TODO - fixme (the noise should only be in the chosen direction)
        action_vector = action_.direction
        next_state = state_.location + action_vector

        # check if in world,, find its id
        if self._domain.in_domain(next_state):
            ns_id = self._domain.find_state(next_state[0], next_state[1])
            if ns_id is not None:
                return [(1.0, ns_id)]
        return [(1.0, state)]

    def trajectory(self, source, target, **kwargs):
        pass

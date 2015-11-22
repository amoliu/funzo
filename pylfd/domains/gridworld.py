"""
GridWorld Domain
"""

from __future__ import division

from collections import Iterable
from matplotlib.patches import Rectangle

import six
import numpy as np

from ..base import Model
from ..models.domain import Domain
from ..models.mdp import MDP, MDPReward, MDPTransition, MDPState, MDPAction


# Cell status
FREE = 'free'
BLOCKED = 'blocked'
TERMINAL = 'terminal'


class GReward(MDPReward):
    """ Grid world MDP reward function """
    def __init__(self, domain):
        super(GReward, self).__init__(domain)

    def __call__(self, state, action):
        state_ = self._domain.S[state]
        if state_.status == TERMINAL:
            return 1.0
        elif state_.status == BLOCKED:
            return -1.0
        else:
            return 0.0

    def __len__():
        return len(self._domain.S)


class GTransition(MDPTransition):
    """ Grirdworld MDP controller """
    def __init__(self, domain, wind=0.2):
        super(GTransition, self).__init__(domain)
        self._wind = wind

    def __call__(self, state, action, **kwargs):
        """ Transition

        Returns
        --------
        A list of all possible next states [(prob, state)]

        """
        state_ = self._domain.S[state]
        action_ = self._domain.A[action]
        p_s = 1.0 - self._wind
        p_f = self._wind / 2.0
        A = self._domain.A.values()
        return [(p_s, self._move(state_, action_)),
                (p_f, self._move(state_, self._right(action_, A))),
                (p_f, self._move(state_, self._left(action_, A)))]

    def _move(self, state, action):
        """ Return the state that results from going in this direction. Stays
        in the same state if action os leading to go outside the world or to
        obstacles

        Returns
        --------
        new_state : int
            Id of the new state after transition (which can be the current
            state, if transition leads to outside of the world)

        """
        new_state = GState((state.cell[0]+action.direction[0],
                           state.cell[1]+action.direction[1]))
        if new_state in self._domain.state_map:
            ns =  self._domain.state_map[new_state]

            # avoid transitions to blocked cells
            if self._domain.S[ns].status == BLOCKED:
                return self._domain.state_map[state]
            return ns

        return self._domain.state_map[state]

    def _heading(self, heading, inc, directions):
        return directions[(directions.index(heading) + inc) % len(directions)]

    def _right(self, heading, directions):
        return self._heading(heading, -1, directions)

    def _left(self, heading, directions):
        return self._heading(heading, +1, directions)


class GState(MDPState):
    """ Gridworld state """
    def __init__(self, cell, status=FREE):
        self.cell = cell
        self.status = status

    def __hash__(self):
        return (self.cell[0], self.cell[1]).__hash__()

    def __eq__(self, other):
        return np.hypot(self.cell[0] - other.cell[0],
                        self.cell[1] - other.cell[1]) < 1e-05

    def __str__(self):
        return '({}, {})'.format(self.cell[0], self.cell[1])

    def __repr__(self):
        return self.__str__()


class GAction(MDPAction):
    """ Grirdworld action """
    def __init__(self, direction):
        self.direction = direction

    def __hash__(self):
        return (self.direction[0], self.direction[1]).__hash__()

    def __eq__(self, other):
        try:
            return all(self.direction == other.direction)
        except Exception:
            return False

    def __str__(self):
        return '[{}, {}]'.format(self.direction[0], self.direction[1])

    def __repr__(self):
        return self.__str__()


class GridWorld(Domain, MDP):
    """ GridWorld domain

    A discrete world with cells (free, obstacles, and goal). The main task
    is to find a path from any start cell to a goal cell.

    """

    def __init__(self, gmap, discount=0.9):
        gr = GReward(domain=self)
        gt = GTransition(domain=self)

        MDP.__init__(self, discount=discount, reward=gr, transition=gt)

        self._gmap = np.asarray(gmap)
        assert self._gmap.ndim == 2, '`gmap` must be a two dimensional array'
        self._initialize(self._gmap)

    def _initialize(self, gmap):
        self._height, self._width = gmap.shape
        self._states = dict()
        self.state_map = dict()  # simple inverse map for transition

        state_id = 0
        for i in range(self._height):
            for j in range(self._width):
                if gmap[i, j] == 1:
                    self._states[state_id] = GState((i, j), BLOCKED)
                if gmap[i, j] == 2:
                    self._states[state_id] = GState((i, j), TERMINAL)
                else:
                    self._states[state_id] = GState((i, j), FREE)

                self.state_map[self._states[state_id]] = state_id
                state_id += 1

        self._actions = {0: GAction((1, 0)), 1: GAction((0, 1)),
                         2: GAction((-1, 0)), 3: GAction((0, -1))}

    @property
    def S(self):
        """ States of the MDP in an indexable container """
        return self._states

    @property
    def A(self):
        """ Actions of the MDP in an indexable container """
        return self._actions

    def terminal(self, state):
        """ Check if a state is terminal"""
        return self.S[state].status == 'terminal'

    def visualize(self, ax, **kwargs):
        if 'show_policy' in kwargs and 'policy' in kwargs:
            print('showing policy')

        ax.imshow(self._gmap, interpolation='nearest',
                  cmap='Paired', vmin=0, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def _show_policy(self, ax, **kwargs):
        pass



#############################################################################

# helper plot utils

# def plot_values(value, ax, mapsize, **kwargs):
#     vmap = np.zeros(shape=mapsize)
#     for k, v in value.items():
#         vmap[k.cell[0], k.cell[1]] = v

#     ax.imshow(vmap, interpolation='nearest', cmap='viridis')
#     ax.set_title('Value function')
#     return ax


# def plot_policy(policy, ax, mapsize, **kwargs):
#     pol = [np.arctan2(a.direction[0], a.direction[1]) for a in policy.values()]
#     pol = np.array(pol).reshape(mapsize)

#     ax.imshow(pol, interpolation='nearest', cmap='viridis')
#     ax.set_title('Policy (direction in radians)')
#     return ax

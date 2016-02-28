"""
GridWorld Domain
"""

from __future__ import division

import warnings

import numpy as np

from six.moves import range
from collections import Iterable
from matplotlib.patches import Rectangle

from .base import Domain

from ..models.mdp import MDP
from ..models.mdp import TabularRewardFunction, LinearRewardFunction
from ..models.mdp import MDPTransition, MDPState, MDPAction

from ..utils.validation import check_random_state


__all__ = [
    'GridWorldMDP',
    'GridWorld',
    'GReward',
    'GRewardLFA',
]

#############################################################################

# Cell status
FREE = 'free'
BLOCKED = 'blocked'
TERMINAL = 'terminal'


class GReward(TabularRewardFunction):
    """ Grid world MDP reward function """
    def __init__(self, domain):
        super(GReward, self).__init__(domain, n_s=len(domain.states))
        R = np.zeros(len(self))
        for s in self._domain.states:
            # state_ = self._domain.states[s]
            if s.status == TERMINAL:
                R[s.id] = 1.0
            elif s.status == BLOCKED:
                R[s.id] = -0.1
            else:
                R[s.id] = -0.001
        self.update_parameters(reward=R)

    def __call__(self, state, action):
        assert state in self._domain.states, 'State does not exist in domain'
        return self._R[state.id]


class GRewardLFA(LinearRewardFunction):
    """ Gridworl reward using linear function approximation """
    def __init__(self, domain, weights):
        super(GRewardLFA, self).__init__(domain, weights)

    def __call__(self, state, action):
        # state_ = self._domain.states[state]
        phi = [self._feature_free(state),
               self._feature_blocked(state),
               self._feature_goal(state)]
        return np.dot(self._weights, phi)

    def __len__(self):
        return 3

    def _feature_goal(self, state):
        """ Check if the agent is at the goal position """
        if state.status == TERMINAL:
            return 1.0
        return 0.0

    def _feature_blocked(self, state):
        """ Check is the agent is in a blocked cell """
        if state.status == BLOCKED:
            return 1.0
        return 0.0

    def _feature_free(self, state):
        """ Check is the agent is in a free cell """
        if state.status == FREE:
            return 1.0
        return 0.0

#############################################################################


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
        # state_ = self._domain.states[state]
        # action_ = self._domain.actions[action]
        p_s = 1.0 - self._wind
        p_f = self._wind / 2.0
        A = self._domain.actions
        return [(p_s, self._move(state, action)),
                (p_f, self._move(state, self._right(action, A))),
                (p_f, self._move(state, self._left(action, A)))]

    def _move(self, state, action):
        """ Return the state that results from going in this direction.

        Stay in the same state if action os leading to go outside the world or
        to obstacles

        Returns
        --------
        new_state : int
            Id of the new state after transition (which can be the current
            state, if transition leads to outside of the world)

        """
        new_coords = (state.cell[0]+action.direction[0],
                      state.cell[1]+action.direction[1])

        new_state = self._domain.find_state(new_coords[0], new_coords[1])
        if new_state is not None:
            # avoid blocked states
            if new_state.status == BLOCKED:
                return state
            return new_state
        return state

        # if new_coords in self._domain.state_map:
        #     ns_id = self._domain.state_map[new_coords]
        #     ns = self._domain.states[ns_id]

        #     # avoid transitions to blocked cells
        #     if ns.status == BLOCKED:
        #         return self._domain.state_map[state.cell]
        #     return ns_id

        # return self._domain.state_map[state.cell]

    def _heading(self, heading, inc, directions):
        return directions[(directions.index(heading) + inc) % len(directions)]

    def _right(self, heading, directions):
        return self._heading(heading, -1, directions)

    def _left(self, heading, directions):
        return self._heading(heading, +1, directions)


#############################################################################


class GState(MDPState):
    """ Gridworld state """
    def __init__(self, state_id, cell, status=FREE):
        super(GState, self).__init__(state_id)
        self.cell = cell
        self.status = status

    def __eq__(self, other):
        return self.id == other.id
        # return np.hypot(self.cell[0] - other.cell[0],
        #                 self.cell[1] - other.cell[1]) < 1e-05

    def __str__(self):
        return '({}, {})'.format(self.cell[0], self.cell[1])

    def __repr__(self):
        return self.__str__()


class GAction(MDPAction):
    """ Grirdworld action """
    def __init__(self, action_id, direction):
        super(GAction, self).__init__(action_id)
        self.direction = direction

    def __eq__(self, other):
        return self.id == other.id
        # try:
        #     return all(self.direction == other.direction)
        # except Exception:
        #     return False

    def __str__(self):
        return '[{}, {}]'.format(self.direction[0], self.direction[1])

    def __repr__(self):
        return self.__str__()


#############################################################################


class GridWorld(Domain):
    """ GridWorld domain

    A discrete world with cells (free, obstacles, and goal). The main task
    is to find a path from any start cell to a goal cell. The start, goal
    and obstacle cells are specified using a matrix or list of lists.

    """

    def __init__(self, gmap):
        self._gmap = np.asarray(gmap)
        assert self._gmap.ndim == 2, '`gmap` must be a two dimensional array'
        self._initialize(np.flipud(self._gmap))

    def _initialize(self, gmap):
        self._height, self._width = gmap.shape
        assert self._height == self._width, 'Only square grids supported'
        self.grid = Grid(nrows=self._width, ncols=self._height)
        self.states = list()
        # self.state_map = dict()  # simple inverse map for transition

        state_id = 0
        for i in range(self.grid.rows):
            for j in range(self.grid.cols):
                if gmap[i, j] == 1:
                    self.grid.block_cell((j, i))
                    self.states.append(GState(state_id, (j, i), BLOCKED))
                elif gmap[i, j] == 2:
                    self.goal = (j, i)
                    self.states.append(GState(state_id, (j, i), TERMINAL))
                else:
                    self.states.append(GState(state_id, (j, i), FREE))

                # self.state_map[(j, i)] = state_id
                state_id += 1

        self.actions = [GAction(0, (1, 0)), GAction(1, (0, 1)),
                        GAction(2, (-1, 0)), GAction(3, (0, -1))]

    def terminal(self, state):
        """ Check if a state is terminal"""
        return self.states[state].status == 'terminal'

    def visualize(self, ax, **kwargs):
        ax = self._setup_visuals(ax)

        if 'policy' in kwargs:
            self.show_policy(ax, kwargs['policy'])

        return ax

    def find_state(self, x, y):
        for s in self.states:
            if np.hypot(s.cell[0] - x, s.cell[1] - y) < 1e-05:
                return s
        return None

    def _setup_visuals(self, ax):
        """ Setup the visual front end for gridworld

        Visuals implemented with y axis flipped upside down to match with
        the array representation in numpy

        """
        cz = 1  # cell size
        for c in self.grid.cells:
            i, j = c[0], c[1]
            s = self.find_state(i, j)
            if s.status == BLOCKED:
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='r', ec='k'))
            elif s.status == TERMINAL:
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='g', ec='k'))
            else:
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='w', ec='k'))

            # if self.grid.blocked((i, j)):
            #     ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
            #                   fc='r', ec='k'))
            # if self.terminal(self.state_map[(i, j)]):
            #     ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
            #                   fc='g', ec='k'))
            # if not self.grid.blocked((i, j)) and\
            #         not self.terminal(self.state_map[(i, j)]):
            #     ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
            #                   fc='w', ec='k'))

        ax.set_xlim([0, self.grid.cols])
        ax.set_ylim([0, self.grid.rows])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def show_policy(self, ax, policy=None):
        """
        Show a policy on the gridworld interface
        """
        if policy is not None:
            assert len(policy) == len(self.states),\
                'Policy not compatible with state space dimensions'
            for s in range(policy.shape[0]):
                a = policy[s]
                if self.actions[int(a)].direction == (1, 0):
                    text = '$\\rightarrow$'
                elif self.actions[int(a)].direction == (0, 1):
                    text = '$\\uparrow$'
                elif self.actions[int(a)].direction == (-1, 0):
                    text = '$\\leftarrow$'
                elif self.actions[int(a)].direction == (0, -1):
                    text = '$\\downarrow$'
                else:
                    text = 'O'
                ss = self.states[s]
                ax.text((ss.cell[0] * 1) + (1 / 2.),
                        (ss.cell[1] * 1) + (1 / 2.),
                        text, ha="center", size=14)
        return ax

    def generate_trajectories(self, policy, num=5, starts=None,
                              random_state=None):
        """ Generate trajectories of varying lengths using a policy """
        assert num > 0, 'Number of trajectories must be greater than zero'
        if starts is not None:
            assert isinstance(starts, Iterable),\
                '{} expects an iterable for *starts*'\
                .format(self.generate_trajectories.__name__)
            num = len(starts)

        controller = GTransition(domain=self)
        trajs = list()
        for i in range(num):
            traj = list()
            if starts is None:
                state = self._pick_random_state(random_state)
            else:
                state = starts[i]

            max_len = self._width * self._height
            while len(traj) < max_len and not self.terminal(state):
                action = policy[state]
                traj.append((state, action))
                next_state = controller(state, action)[0][1]
                state = next_state

            trajs.append(traj)
        return trajs

    def _pick_random_state(self, random_state=None):
        rng = check_random_state(random_state)
        state = rng.randint(len(self.states))
        return state


class GridWorldMDP(MDP):
    """ Grid world MDP representing the decision making process """
    def __init__(self, domain, reward, transition, discount=0.9):
        super(GridWorldMDP, self).__init__(domain,
                                           reward,
                                           transition,
                                           discount)

    @property
    def S(self):
        """ States of the MDP in an indexable container """
        return self._domain.states

    @property
    def A(self):
        """ Actions of the MDP in an indexable container """
        return self._domain.actions

    def actions(self, state):
        return self._domain.actions

    def ix(self, state_id):
        return self._domain.actions[state_id]


#############################################################################

class Grid(object):
    """
    Represents a rectangular grid map. The map consists of
    nrows X ncols coordinates (squares). Some of the squares
    can be blocked (by obstacles).
    """

    def __init__(self, nrows, ncols):
        self._cols = None
        self._rows = None
        self.rows = nrows
        self.cols = ncols
        self._map = np.zeros(shape=(nrows, ncols), dtype=np.int)

    def block_cell(self, c):
        self._map[c[0], c[1]] = 1

    def unblock_cell(self, c):
        self._map[c[0], c[1]] = 0

    def blocked(self, c):
        return self._map[c[0], c[1]] == 1

    def valid_cell(self, cell):
        """ Validity in the general sense, i.e. not outside the grid """
        # and self._map[r][c] == 0):
        r, c = cell[0],  cell[1]
        row_check = 0 <= r <= self._rows - 1
        col_check = 0 <= c <= self._cols - 1
        return row_check and col_check

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = self._check_dim(value, 'rows')

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, value):
        self._cols = self._check_dim(value, 'cols')

    @property
    def cells(self):
        return list((i, j) for i in range(self._rows)
                    for j in range(self._cols))

    @property
    def printme(self):
        for row in range(self._rows):
            for col in range(self._cols):
                print("%s" % ('[X]' if self._map[row, col] == 1 else '[ ]')),
            print('')

    def _check_dim(self, value, dim):
        assert value > 0, '{} must be greater than 0'.format(dim)
        assert isinstance(value, int), '{} must be an integer'.format(dim)
        if value > 100:
            warnings.warn('{} is large, MDP may be slow'.format(dim))
        return value

"""
GridWorld Domain
"""

from __future__ import division

import warnings

import numpy as np

from matplotlib.patches import Rectangle

from ..models.domain import Domain
from ..models.mdp import MDP, MDPReward, MDPTransition, MDPState, MDPAction
from ..utils.validation import check_random_state


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
        reward = -0.001
        if state_.status == TERMINAL:
            reward = 1.0
        elif state_.status == BLOCKED:
            reward = -0.1

        return reward

    def __len__(self):
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
        new_coords = (state.cell[0]+action.direction[0],
                      state.cell[1]+action.direction[1])
        if new_coords in self._domain.state_map:
            ns_id = self._domain.state_map[new_coords]
            ns = self._domain.S[ns_id]

            # avoid transitions to blocked cells
            if ns.status == BLOCKED:
                return self._domain.state_map[state.cell]
            return ns_id

        return self._domain.state_map[state.cell]

    def _heading(self, heading, inc, directions):
        return directions[(directions.index(heading) + inc) % len(directions)]

    def _right(self, heading, directions):
        return self._heading(heading, -1, directions)

    def _left(self, heading, directions):
        return self._heading(heading, +1, directions)


#############################################################################


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


#############################################################################


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
        self._initialize(np.flipud(self._gmap))
        # self._initialize(self._gmap)

    def _initialize(self, gmap):
        self._height, self._width = gmap.shape
        assert self._height == self._width, 'Only square grids supported'
        self.grid = Grid(nrows=self._width, ncols=self._height)
        self._states = dict()
        self.state_map = dict()  # simple inverse map for transition

        state_id = 0
        for i in range(self.grid.rows):
            for j in range(self.grid.cols):
                if gmap[i, j] == 1:
                    self.grid.block_cell((j, i))
                    self._states[state_id] = GState((j, i), BLOCKED)
                elif gmap[i, j] == 2:
                    self.goal = (j, i)
                    self._states[state_id] = GState((j, i), TERMINAL)
                else:
                    self._states[state_id] = GState((j, i), FREE)

                # self.S[state_id] = (j, i)
                self.state_map[(j, i)] = state_id
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

    def actions(self, state):
        return self._actions.keys()

    def terminal(self, state):
        """ Check if a state is terminal"""
        return self.S[state].status == 'terminal'

    def visualize(self, ax, **kwargs):
        ax = self._setup_visuals(ax)

        if 'policy' in kwargs:
            self.show_policy(ax, kwargs['policy'])

        return ax

    def _setup_visuals(self, ax):
        """ Setup the visual front end for gridworld

        Visuals implemented with y axis flipped upside down to match with
        the array representation in numpy

        """
        cz = 1  # cell size
        for c in self.grid.cells:
            i, j = c[0], c[1]
            if self.grid.blocked((i, j)):
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='r', ec='k'))
            if self.terminal(self.state_map[(i, j)]):
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='g', ec='k'))
            if not self.grid.blocked((i, j)) and\
                    not self.terminal(self.state_map[(i, j)]):
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='w', ec='k'))

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
            assert len(policy) == len(self.S),\
                'Policy not compatible with state space dimensions'
            for s in range(policy.shape[0]):
                a = policy[s]
                if self.A[int(a)].direction == (1, 0):
                    text = '$\\rightarrow$'
                elif self.A[int(a)].direction == (0, 1):
                    text = '$\\uparrow$'
                elif self.A[int(a)].direction == (-1, 0):
                    text = '$\\leftarrow$'
                elif self.A[int(a)].direction == (0, -1):
                    text = '$\\downarrow$'
                else:
                    text = 'O'
                ss = self.S[s]
                ax.text((ss.cell[0] * 1) + (1 / 2.),
                        (ss.cell[1] * 1) + (1 / 2.),
                        text, ha="center", size=14)
        return ax

    def generate_trajectories(self, num, policy, random_state=0):
        """ Generate trajectories of varying lengths using a policy """
        assert num > 0, 'Number of trajectories must be greater than zero'
        controller = GTransition(domain=self)

        trajs = list()
        for _ in range(num):
            traj = list()
            state = self._pick_random_state(random_state)
            # traj.append(state)

            while len(traj) < 9 and not self.terminal(state):
                action = policy[state]
                traj.append((state, action))
                next_state = controller(state, action)[0][1]
                # traj.append(next_state)
                state = next_state

            trajs.append(traj)
        return trajs

    def _pick_random_state(self, random_state=0):
        rng = check_random_state(random_state)
        state = rng.randint(len(self.S))
        return state

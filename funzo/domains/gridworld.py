"""
GridWorld Domain

An agent moves in the world using a choice of 5 actions:
(LEFT, RIGHT, UP, DOWN and STAY).

The agent gets a reward that based on the action taken (i.e. depending on the
resulting state). The goal state provides a large positive reward, obstacles
provide large negative reward whiles the remaining states incur a small
constant negative reward (the cost of movement). Additionally, actions that
make the agent go outside of the domain limits incur a negative cost

"""

from __future__ import division

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
OBSTACLE = 'obstacle'
TERMINAL = 'terminal'


class GReward(TabularRewardFunction):
    """ Grid world MDP reward function """
    def __init__(self, domain, rmax=1.0):
        super(GReward, self).__init__(domain,
                                      n_s=len(domain.states),
                                      rmax=rmax)
        R = np.zeros(len(self))
        for s in self._domain.states:
            state_ = self._domain.states[s]
            if state_.status == TERMINAL:
                R[s] = 1.0
            elif state_.status == OBSTACLE:
                R[s] = -10.0
            else:
                R[s] = -0.01
        self.update_parameters(reward=R)

        self._T = GTransition(domain=domain)

    def __call__(self, state, action):
        """ Evaluate reward function """
        if action is None:
            return self._R[state]
        s_p = self._T(state, action)[0][1]
        if s_p == state and action != 4:  # out or domain movements penalty
            return -10.0
        return self._R[s_p]


class GRewardLFA(LinearRewardFunction):
    """ Gridworld reward using linear function approximation """
    def __init__(self, domain, weights, rmax=1.0):
        super(GRewardLFA, self).__init__(domain, weights, rmax=rmax)
        self._T = GTransition(domain=domain)

    def __call__(self, state, action):
        """ Evaluate reward function """
        if action is None:
            return self.__call__(state, 4)
        # get the resulting state to determine if movement occurred
        s_p = self._T(state, action)[0][1]
        if s_p == state and action != 4:  # out or domain movements penalty
            return -10.0

        phi = self.phi(s_p, action)
        return np.dot(self._weights, phi)

    def phi(self, state, action):
        """ Evaluate the reward features for state-action pair """
        state_ = self._domain.states[state]
        phi = np.array([self._feature_free(state_),
                        self._feature_obstacle(state_),
                        self._feature_goal(state_)])
        return phi

    def _feature_goal(self, state):
        """ Check if the agent is at the goal position """
        if state.status == TERMINAL:
            return 1.0
        return 0.0

    def _feature_obstacle(self, state):
        """ Check is the agent is in a OBSTACLE cell """
        if state.status == OBSTACLE:
            return 1.0
        return 0.0

    def _feature_free(self, state):
        """ Check is the agent is in a free cell """
        # Terminal state is always free by definition
        if state.status == FREE or state.status == TERMINAL:
            return 1.0
        return 0.0

#############################################################################


class GTransition(MDPTransition):
    """ Gridworld MDP controller """
    def __init__(self, domain, wind=0.2):
        super(GTransition, self).__init__(domain)
        self._wind = wind

    def __call__(self, state, action, **kwargs):
        """ Transition

        Returns
        --------
        A list of all possible next states [(prob, state)]

        """
        state_ = self._domain.states[state]
        action_ = self._domain.actions[action]
        p_s = 1.0 - self._wind
        p_f = self._wind / 2.0
        A = self._domain.actions.values()
        return [(p_s, self._move(state_, action_)),
                (p_f, self._move(state_, self._right(action_, A))),
                (p_f, self._move(state_, self._left(action_, A)))]

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

        if new_coords in self._domain.state_map:
            return self._domain.state_map[new_coords]

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
    def __init__(self, state_id, cell, status=FREE):
        super(GState, self).__init__(state_id)
        self.cell = cell
        self.status = status

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return (self.cell[0], self.cell[1]).__hash__()

    def __str__(self):
        return '({}, {}, {})'.format(self.cell[0], self.cell[1], self.status)

    def __repr__(self):
        return self.__str__()


class GAction(MDPAction):
    """ Grirdworld action """
    def __init__(self, action_id, direction):
        super(GAction, self).__init__(action_id)
        self.direction = direction

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return (self.direction[0], self.direction[1]).__hash__()

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
        self.states = dict()
        self.state_map = dict()  # simple inverse map for transition

        state_id = 0
        for i in range(self._width):
            for j in range(self._height):
                if gmap[i, j] == 1:
                    self.states[state_id] = GState(state_id, (j, i), OBSTACLE)
                elif gmap[i, j] == 2:
                    self.goal = (j, i)
                    self.states[state_id] = GState(state_id, (j, i), TERMINAL)
                else:
                    self.states[state_id] = GState(state_id, (j, i), FREE)

                self.state_map[(j, i)] = state_id
                state_id += 1

        self.actions = {
            0: GAction(0, (1, 0)),
            1: GAction(1, (0, 1)),
            2: GAction(2, (-1, 0)),
            3: GAction(3, (0, -1)),
            4: GAction(4, (0, 0))
        }

    def terminal(self, state):
        """ Check if a state is terminal"""
        return self.states[state].status == 'terminal'

    def visualize(self, ax, **kwargs):
        ax = self._setup_visuals(ax)

        if 'policy' in kwargs:
            self.show_policy(ax, kwargs['policy'])

        return ax

    def in_domain(self, x, y):
        """ Check if a cell is in the domain """
        return (x, y) in self.state_map

    def _setup_visuals(self, ax):
        """ Setup the visual front end for gridworld

        Visuals implemented with y axis flipped upside down to match with
        the array representation in numpy

        """
        cz = 1  # cell size in matplotlib units
        for c in self.states:
            s = self.states[c]
            i, j = s.cell[0], s.cell[1]
            if s.status == OBSTACLE:
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='#b91d47', ec='#2b5797', alpha=0.7))
            elif s.status == TERMINAL:
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='#00a300', ec='#2b5797', alpha=0.7))
            else:
                ax.add_artist(Rectangle((i * cz, j * cz), cz, cz,
                              fc='w', ec='#2b5797'))

        ax.set_xlim([0, self._width])
        ax.set_ylim([0, self._height])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def show_policy(self, ax, policy=None):
        """ Show a policy on the gridworld interface """
        if policy is not None:
            assert len(policy) == len(self.states),\
                'Policy not compatible with state space dimensions'
            for s in range(policy.shape[0]):
                a = policy[s]
                if self.actions[a].direction == (1, 0):
                    text = '$\\rightarrow$'
                elif self.actions[a].direction == (0, 1):
                    text = '$\\uparrow$'
                elif self.actions[a].direction == (-1, 0):
                    text = '$\\leftarrow$'
                elif self.actions[a].direction == (0, -1):
                    text = '$\\downarrow$'
                else:
                    text = 'S'
                ss = self.states[s]
                ax.text((ss.cell[0] * 1) + (1 / 2.),
                        (ss.cell[1] * 1) + (1 / 2.3),
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

            if self.terminal(state):
                traj.append((state, 4))  # FIXME

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
        return self._domain.states.keys()

    @property
    def A(self):
        """ Actions of the MDP in an indexable container """
        return self._domain.actions.keys()

    def actions(self, state):
        return self._domain.actions.keys()

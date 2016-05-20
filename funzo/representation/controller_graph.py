from __future__ import division, absolute_import

import json

import numpy as np


from .state_graph import StateGraph


class ControllerGraph(object):
    """ A Controller Graph

    A graph based representation of continuous state MDPs by sampling a few
    states and connecting them using local controllers (which can be seen as
    Markov options). Generates a very sparse representation which is efficient
    and also allows for task constraints to be modeled directly into the MDP
    action space via these local controllers.

    """
    def __init__(self, params, controller, domain=None):
        super(ControllerGraph, self).__init__()
        self._controller = controller


class CGParameters(object):
    """ ControllerGraph parameters

    Uses json encoding for persistence, so as to allow different domain
    configurations.

    Attributes
    ------------
    n_expand : int
        Number of nodes to expand at every iteration
    n_new : int
        Number of new nodes to sample around the expansion node (represents
        the number of trials)
    n_add : int
        Number of nodes to add to the graph from the set of sampled nodes.
        Allows sampling and ranking to select only top-K nodes
    radius : float
        Radius around a node to sample new nodes from (effectively controls
        the extent of running the local controller)
    exp_thresh : float
        Threshold on the value of a node to select it for expansion
    max_traj_len : int
        Maximum allowed length of trajectories (policies) in the CG
    p_best : float
        Probability of expanding from the set of nodes which are part of the
        current best policies to the goal at every iteration
    max_samples : int
        Maximum number of samples/nodes for the CG. Used for termination in the
        building of the graph
    max_edges : int
        Maximum number of edges allowed. Controls sparsity of the graph and
        also helps to encourage exploration
    init_type : string
        CG initialization type, either: random or trajectory
            * random -- sample initial set of nodes uniformly from the world
            * trajectory -- sample the initial set of nodes from provided
            expert demonstration trajectories.
    max_cost : float
        TODO
    conc_scale : None
        TODO
    tmin : float
        Lowe, time limit for running the local controller. Tempered with
        iterations.
    tmax : float
        Upper time limit for running the local controller. Tempered with
        iterations.

    """

    _PARAMS = [
        'n_expand',
        'n_new',
        'n_add',
        'radius',
        'exp_thresh',
        'max_traj_len',
        'p_best',
        'max_samples',
        'max_edges',
        'init_type',
        'max_cost',
        'conc_scale',
        'tmin',
        'tmax',
    ]

    def __init__(self, **kwargs):
        self.n_expand = kwargs.pop('n_expand', 1)
        self.n_new = kwargs.pop('n_new', 20)
        self.n_add = kwargs.pop('n_add', 1)
        self.radius = kwargs.pop('radius', 1.8)
        self.exp_thresh = kwargs.pop('exp_thresh', 1.2)
        self.max_traj_len = kwargs.pop('max_traj_len', 500)
        self.p_best = kwargs.pop('p_best', 0.4)
        self.max_samples = kwargs.pop('max_samples', 100)
        self.max_edges = kwargs.pop('max_edges', 360)
        self.init_type = kwargs.pop('init_type', 'random')
        self.max_cost = kwargs.pop('max_cost', 1000)
        self.conc_scale = kwargs.pop('conc_scale', 1)
        self.tmin = kwargs.pop('tmin', (0.45, 2.4))
        self.tmax = kwargs.pop('tmax', (3.6, 7.2))

    def load(self, json_file):
        """ Load parameters from a json file """
        with open(json_file, 'r') as f:
            jdata = json.load(f)
            for k, v in jdata.items():
                self.__dict__[k] = v

    def save(self, filename):
        """ Save the parameters to file """
        with open(filename, 'w') as f:
            json.dump(self._to_dict(), f, indent=4, sort_keys=True)

    def __repr__(self):
        return self._to_dict()

    def __str__(self):
        d = self._to_dict()
        return ''.join('{}: {}\n'.format(k, v) for k, v in d.items())

    def _to_dict(self):
        return self.__dict__

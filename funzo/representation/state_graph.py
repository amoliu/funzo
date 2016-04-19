from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import warnings
import pickle

import networkx as nx

from six.moves import filter
from numpy import asarray, sqrt


__all__ = ['StateGraph']


class StateGraph(object):
    """ Generic state graph suited for MDPs

    The state graph encapsulates a flexible representation for an MDP which
    affords use of task specific constraints as well as temporally extended
    actions (in the sense of hierarchical reinforcement learning, options)

    Parameters
    -----------
    state_dim : int
        The dimensional of the state space used in the graph

    Attributes
    ------------
    _graph : :class:`networkx.DiGraph` object
        The underlying graph using ``networkx``
    _node_attrs : tuple or str
        Node attributes used in the graph
    _edge_attrs : tuple of str
        Attribute types of the edges in the graph
    _state_dim : int
        The dimensional of the state space used in the graph

    """

    _node_attrs = ('data', 'cost', 'priority', 'Q', 'V', 'pi', 'type')
    _edge_attrs = ('source', 'target', 'duration', 'reward', 'phi', 'traj')

    def __init__(self, state_dim):
        self._graph = nx.DiGraph()

        if state_dim <= 0:
            raise ValueError('State dimension must be greater than 0')
        self._state_dim = state_dim

    def clear(self):
        """ Reset the graph """
        self.G.clear()

    def add_node(self, nid, data, cost, priority, Q, V, pi, ntype):
        """ Add a new node to the graph """
        data = asarray(data)
        if len(data) != self._state_dim:
            raise ValueError('Expecting a {}-dim state vector for\
                node'.format(self._state_dim))

        if nid not in self.G:
            self.G.add_node(nid, data=data, cost=cost, priority=priority,
                            Q=Q, V=V, pi=pi, type=ntype)
        else:
            warnings.warn('Node already exits in the graph, not added')

    def add_edge(self, source, target, duration, reward, phi, traj):
        """ Add a new edge into the graph """
        if duration < 0.0:
            raise ValueError('Duration arguiment must be positive, >= 0')
        phi = asarray(phi)
        traj = asarray(traj)
        if traj.ndim != 2:
            raise ValueError('Expecting a 2-dim dim trajectory')

        if source == target:
            warnings.warn('source: {} and target: {} nodes are the same'.
                          format(source, target))

        elif not self.G.has_edge(source, target):
            self.G.add_edge(source, target, duration=duration,
                            reward=reward, phi=phi, traj=traj)
        else:
            warnings.warn('Edge ({}--{}) already exists in the graph'
                          .format(source, target))

    def remove_edge(self, source, target):
        """ Remove an edge from the graph """
        if source == target:
            warnings.warn('source: {} and target: {} nodes are the same'.
                          format(source, target))

        self.G.remove_edge(source, target)

    def remove_node(self, node):
        """ Remove a node from the graph """
        self.G.remove_node(node)

    def edge_exists(self, source, target):
        """ Check if an edge already exists in the graph """
        return self.G.has_edge(source, target)

    def gna(self, node_id, attribute):
        """ Get a single attribute of a single node

        Parameters
        ------------
        node_id : int
        attribute : string

        """
        self._check_node_attributes(node_id, attribute)
        return self.G.node[node_id][attribute]

    def sna(self, node_id, attribute, value):
        """ Set a single attribute of a node

        Parameters
        ------------
        node_id : int
        attribute : string
        value : any

        """
        self._check_node_attributes(node_id, attribute)
        self.G.node[node_id][attribute] = value

    def gea(self, source, target, attribute):
        """ Get a single attribute of a single edge """
        self._check_edge_attributes(source, target, attribute)
        return self.G.edge[source][target][attribute]

    def sea(self, source, target, attribute, value):
        """ Set a single attribute of a edge between source and target """
        self._check_edge_attributes(source, target, attribute)
        self.G.edge[source][target][attribute] = value

    def find_neighbors_data(self, c, distance, metric=None):
        """ Find node neighbors based on distance between `data` attribute

        Parameters
        -----------
        c : array-like, shape = N,
            `data` array to search around
        distance: float
            Maximum range for inclusion the returned neighbors list
        metric : callable, optional (default : None)
            Metric function for deciding 'closeness' wrt to `data` attribute
            If `None`, Euclidean distance will be used

        Returns
        -------
        neighbors : list of int
            List of node ids in the "neighborhood"

        Notes
        ------
        Includes the query node in the result

        """
        m = metric
        if metric is None:
            m = eud

        neighbors = filter(lambda n: m(self.gna(n, 'data'), c) <= distance,
                           self.G.nodes())
        return list(neighbors)

    def find_neighbors_range(self, nid, distance):
        """ Find neighboring nodes within a distance

        Parameters
        -----------
        nid : int
            Node id for the query node
        distance: float
            Maximum range for inclusion the returned neighbors list

        Returns
        -------
        neighbors : list of int
            List of node ids in the "neighborhood"

        Notes
        ------
        Includes the query node in the result

        """
        cn = self.gna(nid, 'data')
        return self.find_neighbors_data(cn, distance, None)

    def find_neighbors_k(self, nid, k):
        """ Find k nearest neighbors based on Euclidean distance

        The Euclidean distance is computed based on the `data` attribute

        Parameters
        -----------
        nid : int
            Node id for the query node
        k: int
            Maximum number of nodes to return

        Returns
        -------
        neighbors : list of int
            List of node ids in the "neighborhood"

        """
        serch_set = set(self.G.nodes()) - {nid}
        cn = self.gna(nid, 'data')
        distances = {n: eud(self.gna(n, 'data'), cn) for n in serch_set}
        sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])
        k_neighbors = sorted_neighbors[:k]
        return list(n[0] for n in k_neighbors)

    def neighbors(self, nid):
        """ Get the connected node neighbors """
        return self.G.neighbors(nid)

    def edges(self, nid):
        """ Return the edges of a node """
        return self.G.edges(nid)

    def out_edges(self, nid):
        """ Return the outgoing edges of a node """
        return self.G.out_edges(nid)

    def filter_nodes_by_type(self, ntype):
        """ Filter nodes by node type """
        sns = filter(lambda n: self.gna(n, 'type') == ntype, self.nodes)
        return list(sns)

    def search_path(self, source, target):
        """ Search for a path from ``source`` to ``target`` using A*"""
        def metric(a, b):
            if self.edge_exists(source, target):
                return -1*self.gea(source, target, 'reward')
            return 1000
        path = nx.astar_path(self.G, source, target, heuristic=metric)
        return path

    def get_signal(self, name):
        """ Retrieve a graph signal from the nodes

        The signals correspond to the node attributes in the graph. For Q
        values, the signal is a list of lists, each of varying lengths since
        the number of edges vary per node.

        Parameters
        -----------
        name : str
            Name of signal to retrieve

        Returns
        -------
        signal : array-like
            1D array for Cost, V, and policy; and a list of lists for Q

        """
        if name not in self._node_attrs:
            raise IndexError('Invalid signal name')
        return [self.gna(n, name) for n in self.nodes]

    def save_graph(self, filename):
        """ Save the graph to file """
        with open(filename, 'wb') as f:
            pickle.dump(self._graph, f)

    def load_graph(self, filename):
        """ Load a graph from file """
        with open(filename, 'rb') as f:
            self._graph = pickle.load(f)

    def save_svg(self):
        raise NotImplementedError('Not implemented')

    def _check_node_attributes(self, node_id, attribute):
        assert attribute in self._node_attrs,\
            'Attribute [{}] is invalid | Expected:{}'\
            .format(attribute, self._node_attrs)
        assert node_id in self.nodes, \
            'Node ({}) not in the graph'.format(node_id)

    def _check_edge_attributes(self, source, target, attribute):
        assert attribute in self._edge_attrs, \
            'Attribute [{}] is invalid | Expected:{}'\
            .format(attribute, self._edge_attrs)
        assert self.G.has_edge(source, target),\
            'Edge [{}-{}] does not exist in the graph'.format(source, target)

    @property
    def G(self):
        return self._graph

    @property
    def nodes(self):
        return self.G.nodes()

    @property
    def nodes_data(self):
        return self.G.nodes(data=True)

    @property
    def all_edges(self):
        return self.G.edges()

    @property
    def transition_matrix(self):
        """ Get the transition matrix T(s, a, s')

        Obtained from the adjacency matrix of the underlying graph

        """
        return nx.adjacency_matrix(self.G).todense()


def eud(data1, data2):
    return sqrt((data1[0]-data2[0])**2 + (data1[1]-data2[1])**2)

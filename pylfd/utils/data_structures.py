"""
Base data structres used throughout the package

"""

from __future__ import division, absolute_import

from collections import MutableMapping, Hashable, Iterable
from operator import itemgetter

import numpy as np


class ValueFunction(MutableMapping, dict):
    """ ValueFunction

    Value function represented as a mappable container from states to their
    values, i.e.

    .. math::
        V: \mathcal{S} \longrightarrow \mathbb{R}

    """
    def __init__(self, states):
        assert isinstance(states, Iterable), '*states* must be iterable'
        for s in states:
            self.__setitem__(self._check_key(s), 0.0)

    def __getitem__(self, key):
        return dict.__getitem__(self, self._check_key(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, self._check_key(key), value)

    def keys(self):
        return dict.keys(self)

    def values(self):
        return dict.values(self)

    def __iter__(self):
        return dict.__iter__(self)

    def _check_key(self, key):
        assert isinstance(key, Hashable), \
            '{} must be a hashable object'.format(key)
        return key


class Policy(MutableMapping, dict):
    """ Policy

    A policy is a mapping from states to actions

    .. math::
        \pi: \mathcal{S} \longrightarrow \mathbb{A}

    """
    def __init__(self, states, actions):
        assert isinstance(states, Iterable), '*states* must be iterable'
        assert isinstance(actions, Iterable), '*actions* must be iterable'
        for s in states:
            a = np.random.randint(len(actions))
            self.__setitem__(self._check_domain(s), actions[a])

    def __getitem__(self, key):
        return dict.__getitem__(self, self._check_domain(key))

    def __setitem__(self, key, value):
        action = self._check_domain(value)
        dict.__setitem__(self, self._check_domain(key), action)

    def keys(self):
        return dict.keys(self)

    def values(self):
        return dict.values(self)

    def __iter__(self):
        return dict.__iter__(self)

    def _check_domain(self, key):
        assert isinstance(key, Hashable), \
            '{} must be a hashable object'.format(key)
        return key


class QFunction(MutableMapping, dict):
    """ Action-value function (Q function)

    The Q function is represented as a table, accessible by key values pairs
    of states and actions, i.e. (state, action) -> R

    """
    def __init__(self, states, actions):
        assert isinstance(states, Iterable), '*states* must be iterable'
        assert isinstance(actions, Iterable), '*actions* must be iterable'
        for s in states:
            for a in actions:
                self.__setitem__((self._check_domain(s),
                                  self._check_domain(a)), np.random.random())

    def __getitem__(self, key):
        self._check_domain(key[0])
        self._check_domain(key[1])
        return dict.__getitem__(self, (key[0], key[1]))

    def __setitem__(self, key, value):
        self._check_domain(key[0])
        self._check_domain(key[1])
        dict.__setitem__(self, (key[0], key[1]), value)

    def keys(self):
        return dict.keys(self)

    def values(self):
        return dict.values(self)

    @property
    def policy(self):
        """ Extract the policy via argmax """
        pi_hat = Policy(list(sa[0] for sa in self.keys()),
                        list(sa[1] for sa in self.keys()))
        for s_a in self.keys():
            s = s_a[0]
            # find all the occurences of state s
            actions = {p[1] : self[p] for p in self if p[0]==s}
            a = list(sorted(actions, key=actions.get))[-1]
            pi_hat[s] = a

        return pi_hat

    def __iter__(self):
        return dict.__iter__(self)

    def _check_domain(self, key):
        assert isinstance(key, Hashable), \
            '{} must be a hashable object'.format(key)
        return key

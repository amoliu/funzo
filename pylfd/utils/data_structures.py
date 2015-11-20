"""
Base data structres used throughout the package

"""

from __future__ import division, absolute_import

from collections import MutableMapping, Hashable, Iterable

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

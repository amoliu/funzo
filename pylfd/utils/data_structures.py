"""
Base data structres used throughout the package

"""

from __future__ import division, absolute_import

from collections import MutableMapping, Hashable


class ValueFunction(MutableMapping):
    """ ValueFunction

    Value function represented as a mappable container from states to their
    values, i.e.

    .. math::
        V: \mathcal{S} \longrightarrow \mathbb{R}

    """
    def __init__(self, *arg, **kwargs):
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[self._check_key(key)]

    def __setitem__(self, key, value):
        self.store[self._check_key(key)] = value

    def __delitem__(self, key):
        del self.store[self._check_key(key)]

    def _check_key(self, key):
        assert isinstance(key, Hashable), \
            '{} must be a hashable object'.format(key)
        return key

    def argmax(self):
        return None

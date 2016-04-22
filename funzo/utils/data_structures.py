"""
Utility data structures
"""

from __future__ import division, absolute_import

import os
import time
import warnings

try:
    import h5py
except ImportError:
    warnings.warn('h5py needed to use this module')

from collections import Iterable


class Trace(object):

    """ Iterative data store

    Data structure for storing progress logs of various iterative algorithms

    Parameters
    -----------
    save_interval : int, optional (default: 100)
        No. of iterations before a periodic save of current data
    variables : list of string or str, optional (default: None)
        Names of all the variables stored, a single variable name

     Attributes
    ------------
    _save_interval : int
        No. of iterations before a periodic save of current data. Zero means
        no periodic saves
    _variables : list of string
        Names of all the variables stored

    """

    def __init__(self, variables=None, save_interval=100):
        if save_interval < 0:
            raise ValueError('Saving interval must be >= 0')
        if save_interval > 1000:
            warnings.warn('Very large saving interval could result in\
                          loss of data in a crash')

        self._save_interval = save_interval
        self._data = dict()
        self._vars = None

        if variables is not None:
            if not isinstance(variables, Iterable):
                raise ValueError('*variables* must be an iterable container')

            if not all([isinstance(v, str) for v in variables]):
                raise TypeError('All *variables* must be strings type')

            if isinstance(variables, str):
                warnings.warn('Single variable assigned. Assuming a single\
                               element trace store')
                variables = list(variables)

            self._vars = variables
            for v in self._vars:
                self._data[v] = list()

        self._old_save = None
        self._iter = 0

    def record(self, **entry):
        """ Record new data into the trace

        Allows asynchronous recording of only select variables. Adding an index
        variable then helps to later align data.

        """
        for v in entry:
            if v in self.vars:
                self._data[v].append(entry[v])
            else:
                raise KeyError('{} not a variable name'.format(v))

        self._iter += 1

        periodic_save = self._save_interval > 0
        if periodic_save:
            if self._iter > 0 and self._iter % self._save_interval == 0:
                self.save('trace')

    def save(self, filename='trace', final=False):
        """ Save trace as an HDF5 file with groups """
        if final:
            saved_name = '{}.hdf5'.format(filename)
        else:
            saved_name = '{}_{}.hdf5'.format(filename, time_string())
        f = h5py.File(saved_name, mode='w')
        for key in self._data:
            f[key] = self._data[key]
        f.close()

        # if self._old_save is not None:
        #     os.remove(self._old_save)
        # self._old_save = saved_name

        return saved_name

    def load(self, filename):
        """ Load a trace from file """
        self._vars = list()
        self._data = dict()

        f = h5py.File(filename, mode='r')
        for key in f:
            self._vars.append(key)
            self._data[key] = f[key][:]
        f.close()

    def add_vars(self, new_vars):
        for v in new_vars:
            self._data[v] = list()
            self._vars.append(v)

    @property
    def vars(self):
        """ Variables in the trace data store """
        return self._vars

    def plot(self, axes):
        """ Plot the trace to visually inspect convergence """
        raise NotImplementedError('Not yet implemented')

    def __getitem__(self, item):
        if item not in self:
            raise ValueError('Invalid key: {} not available'.format(item))
        return self._data[item]

    def __contains__(self, key):
        return key in self._vars if self._vars is not None else False


def time_string():
    """ Get a formatted string representation of the current time """
    return time.strftime("%d-%m-%Y_%H:%M:%S")

"""
Data structure helpers
========================

1. Trace -- MCMC sampling logs

"""

from __future__ import division, absolute_import

import os
import h5py
import time
import warnings


class Trace(object):

    """ Reward learning trace containing relevant data about the progress """

    def __init__(self, save_interval=100):
        if save_interval < 0:
            raise ValueError('Saving interval must be > 0')
        if save_interval > 1000:
            warnings.warn('Very large saving interval could result in\
                          loss of data in a crash')

        self._save_interval = save_interval
        self._vars = ['step', 'r', 'sample', 'a_ratio', 'Q_r', 'log_p']

        self.data = dict()
        for v in self._vars:
            self.data[v] = list()

        self._old_save = None

    def record(self, step, r, sample, a_ratio, Q, log_p):
        if len(r) != len(sample):
            raise ValueError('Reward and sample must have same dim')
        if step <= 0:
            raise ValueError('Sample step cannot be < 0')

        self.data['r'].append(r)
        self.data['step'].append(step)
        self.data['sample'].append(sample)
        self.data['a_ratio'].append(a_ratio)
        self.data['Q_r'].append(Q)
        self.data['log_p'].append(log_p)

        if step > 0 and step % self._save_interval == 0:
            self.save('trace')

    def save(self, filename='trace'):
        """ Save trace as an HDF5 file with groups """
        saved_name = '{}_{}.hdf5'.format(filename, time_string())
        f = h5py.File(saved_name, 'w')
        for key in self.data:
            f[key] = self.data[key]
        f.close()

        # if self._old_save is not None:
        #     os.remove(self._old_save)
        # self._old_save = saved_name

        return saved_name

    @property
    def vars(self):
        return self._vars

    def plot(self, axes):
        """ Plot the trace to visually inspect convergence """
        raise NotImplementedError('Not yet implemented')

    def __getitem__(self, item):
        if item not in self.data:
            raise ValueError('Invalid key: {} not available'.format(item))
        return self.data[item]


def time_string():
    """ Get a formatted string representation of the current time """
    return time.strftime("%d-%m-%Y_%H:%M:%S")

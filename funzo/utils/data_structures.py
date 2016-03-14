"""
Data structure helpers
========================

1. Trace -- MCMC sampling logs

"""

from __future__ import division, absolute_import

import h5py
import time
import warnings


class Trace(object):

    """ MCMC sampling trace """

    def __init__(self, save_interval=100):
        if save_interval < 0:
            raise ValueError('Saving interval must be > 0')
        if save_interval > 1000:
            warnings.warn('Very large saving interval could result in\
                          loss of data in a crash')

        self._save_interval = save_interval

        self.data = dict()
        self.data['r'] = None
        self.data['step'] = list()
        self.data['sample'] = list()
        self.data['accept'] = list()
        self.data['Q_r'] = list()
        self.data['logp'] = list()

    def record(self, r, sample, step, accept, Q, logp):
        if len(r) != len(sample):
            raise ValueError('Reward and sample must have same dim')
        if step <= 0:
            raise ValueError('Sample step cannot be < 0')

        self.data['r'] = r
        self.data['step'].append(step)
        self.data['sample'].append(sample)
        self.data['accept'].append(accept)
        self.data['Q_r'].append(Q)
        self.data['logp'].append(logp)

        if step > 0 and step % self._save_interval == 0:
            self.save('trace')

    def save(self, filename='trace'):
        """ Save trace as an HDF5 file with groups """
        saved_name = '{}_{}.hdf5'.format(filename, time_string())
        f = h5py.File(saved_name, 'w')
        for key in self.data:
            f[key] = self.data[key]
        f.close()
        return saved_name

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

"""
Data structure helpers
========================

1. Trace -- MCMC sampling logs

"""

from __future__ import division, absolute_import

import h5py
import time


class Trace(object):

    """ MCMC sampling trace """

    def __init__(self, save_interval=100):
        self.data = dict()
        self.data['r'] = None
        self.data['step'] = list()
        self.data['sample'] = list()
        self.data['accept'] = list()
        self.data['Q_r'] = list()
        self.data['logp'] = list()

    def record(self, r, step, sample, accept, Q, logp):
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
        file_name = '{}_{}.hdf5'.format(filename, time_string())
        f = h5py.File(file_name, 'w')
        for key in self.data:
            f[key] = self.data[key]
        f.close()

    def plot(self, axes):
        """ Plot the trace to visually inspect convergence """
        raise NotImplementedError('Not yet implemented')


def time_string():
    """ Get a formatted string representation of the current time """
    return time.strftime("%d-%m-%Y_%H:%M:%S")

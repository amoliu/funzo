
import h5py
import os

import numpy as np

from nose.tools import assert_raises, assert_is_instance
from numpy.testing import assert_equal

from funzo.utils.data_structures import Trace


def test_trace_init():
    """ Test for proper instantiated of a Trace """
    t = Trace(variables=['step'])
    assert len(t['step']) == 0

    t2 = Trace(variables=['step'], save_interval=0)  # check no periodic
    assert len(t['step']) == 0

    assert_is_instance(t, Trace)
    assert_is_instance(t2, Trace)

    assert_raises(ValueError, t.__init__,
        variables=['step'], save_interval=-1)
    #assert_raises(ValueError, t.__init__, variables='step')
    assert_raises(TypeError, t.__init__, variables=['step', 4])


def test_Trace_record():
    """ Test record function of Trace """
    t = Trace(variables=['step', 'r'])
    t.record(step=1, r=[1.0, 2.0])
    t.record(step=2, r=[15.0, 2.0])
    t.record(step=3, r=[4.0, 2.0])
    assert len(t['step']) == 3
    assert len(t['r']) == 3

    t.record(step=4)
    assert len(t['r']) == 3
    assert len(t['step']) == 4

    # check assertions in the code due to invalid mix of values
    assert_raises(KeyError, t.record, step=1, g=[1.0, 2.0, 3.0])


def test_trace_save():
    """ Test saving Trace """
    t = Trace(variables=['step', 'r', 'q'])

    np.random.seed(42)

    qq = np.random.rand(5)

    t.record(step=1, r=[1.0, 2.0], q=qq)
    fname = t.save('trace')

    f = h5py.File(fname, 'r')
    for k in t.vars:
        assert k in f

    assert_equal(f['r'][0], [1.0, 2.0])
    assert_equal(f['q'][0], qq)

    os.remove(fname)


def test_trace_load():
    """ Test loading Trace """
    t = Trace(variables=['step', 'r', 'q'])

    np.random.seed(42)
    qq = np.random.rand(5)
    t.record(step=1, r=[1.0, 2.0], q=qq)
    fname = t.save('trace', final=True)

    # load in a new trace instance
    t2 = Trace()
    t2.load(fname)
    assert 'r' in t2.vars
    assert 'q' in t2.vars
    assert len(t2['r']) == 1
    assert_equal(t2['r'][0], [1.0, 2.0])
    assert_equal(t2['q'][0], qq)

    os.remove(fname)


def test_trace_getitem():
    """ Test __getitem__ (indexing) function of Trace """
    t = Trace(variables=['r'])
    assert_raises(ValueError, t.__getitem__, 'V')


def test_trace_membership():
    """ Test __contains__ (membership) function of Trace """
    t = Trace(variables=['r'])
    assert 'r' in t
    assert 'x' not in t
    assert 10 not in t

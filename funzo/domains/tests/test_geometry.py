
import numpy as np

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from funzo.domains.geometry import discretize_space
from funzo.domains.geometry import edist
from funzo.domains.geometry import distance_to_segment


def test_distance_to_segment():
    # test points
    x1 = np.array([2.0, 2.0])  # collinear inside
    x2 = np.array([4.0, 0.0])  # collinear outside
    x3 = np.array([4.0, 1.0])  # outside not collinear
    # x4 = np.array([0.0, -1.0])  # inside not collinear
    x5 = np.array([1.0, 2.0])  # inside not collinear
    x6 = np.array([2.7, 2.7])  # inside not collinear

    # line
    ls = np.array([1.0, 3.0])
    le = np.array([3.0, 1.0])

    assert_equal(distance_to_segment(x1, ls, le)[1], True)
    assert_equal(distance_to_segment(x1, ls, le)[0], 0.0)

    assert_equal(distance_to_segment(x2, ls, le)[1], False)
    assert_equal(distance_to_segment(x3, ls, le)[1], False)

    assert_equal(distance_to_segment(x6, ls, le)[1], True)
    # assert_equal(distance_to_segment(x4, ls, le)[1], True)
    assert_equal(distance_to_segment(x5, ls, le)[1], True)


def test_edist():
    # toy data
    pose1 = np.array([2, 2])
    pose2 = np.array([12, 2])
    assert_equal(10, edist(pose1, pose2))
    assert_equal(10, edist(pose1.tolist(), pose2.tolist()))
    assert_equal(10, edist((2, 2), (12, 2)))


def test_discretize_space():
    x = np.arange(0.0, 10.0, 1.0)
    x_d = discretize_space((0.0, 10.0, 1.0))
    assert_array_equal(x, x_d[0])


if __name__ == '__main__':
    import nose
    nose.runmodule()

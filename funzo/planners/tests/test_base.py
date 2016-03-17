
from nose.tools import assert_raises

from funzo.planners.base import Planner


def test_planner_interface():
    """ Test to check the planner interface cannot be instantiated """
    assert_raises(TypeError, Planner)

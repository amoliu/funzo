

from nose.tools import assert_raises

from funzo.models.mdp import MDPState
from funzo.models.mdp import MDPAction
from funzo.models.mdp import RewardFunction
from funzo.models.mdp import LinearRewardFunction
from funzo.models.mdp import TabularRewardFunction
from funzo.models.mdp import MDPTransition
from funzo.models.mdp import MDP


def test_state_init():
    # check for non-instantiation of abstract class/interface
    assert_raises(TypeError, MDPState, 0)
    assert_raises(TypeError, MDPAction, 0)
    assert_raises(TypeError, RewardFunction, None)
    assert_raises(TypeError, MDPTransition, None)
    assert_raises(TypeError, MDP, None, None, None)
    assert_raises(TypeError, TabularRewardFunction, None, 0)
    assert_raises(TypeError, LinearRewardFunction, None, (0, 0))


def test_f2():
    assert 1


import pytest

from funzo.models.mdp import MDPState
from funzo.models.mdp import MDPAction
from funzo.models.mdp import RewardFunction
from funzo.models.mdp import LinearRewardFunction
from funzo.models.mdp import TabularRewardFunction
from funzo.models.mdp import MDPTransition
from funzo.models.mdp import MDP


def test_state_init():
    # check for non-instantiation of abstract class/interface
    with pytest.raises(TypeError):
        MDPState(0)


def test_f2():
    assert 1

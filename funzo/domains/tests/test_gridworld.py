

import pytest

from funzo.domains.gridworld import GState, GAction, GTransition, GReward
from funzo.domains.gridworld import GridWorld, Grid


def test_grid():
    tmap = [[0, 0, 0], [0, 1, 0], [0, 0, 2]]
    grid = Grid(3, 3)

    assert 1


def test_pass():
    assert 1

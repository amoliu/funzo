"""
Planning in an MDP represented using CGs
"""

from __future__ import division


import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

from funzo.domains.social_navigation import SocialNavigationWorld
from funzo.domains.social_navigation import CGSocialNavigationMDP

from funzo.domains.social_navigation import LinearController

from funzo.domains.gridworld import GRewardLFA


def main():
    with SocialNavigationWorld(x=0, y=0, w=10, h=10, goal=(6, 6, 0),
                               persons=None, groups=None) as world:
        R = None
        C = LinearController(resolution=0.1)

        smdp = CGSocialNavigationMDP(R, C, 0.9)
        smdp.setup_CG(None, [(1,1,1), (3,3,3,), (4,1,2)])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    ax = world.visualize(ax, graph=smdp.graph)

    plt.show()


if __name__ == '__main__':
    main()

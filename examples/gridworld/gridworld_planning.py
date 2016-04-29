from __future__ import division

import argparse

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np

from funzo.domains.gridworld import GridWorld, GridWorldMDP
from funzo.domains.gridworld import GReward, GRewardLFA, GTransition
from funzo.planners.dp import PolicyIteration, ValueIteration


def main(map_name, planner):
    gmap = np.loadtxt(map_name)
    w, h = gmap.shape

    with GridWorld(gmap=gmap) as world:
        R = GReward(ns=w * h)
        T = GTransition(wind=0.1)
        g_mdp = GridWorldMDP(reward=R, transition=T, discount=0.99)

        # ------------------------
        mdp_planner = PolicyIteration(max_iter=200, random_state=None)
        if planner == 'VI':
            mdp_planner = ValueIteration(verbose=2)

        res = mdp_planner.solve(g_mdp)
        V = res['V']
        print('Policy: ', res['pi'])


    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = world.visualize(ax, policy=res['pi'])

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='inferno', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.grid(False)
    plt.title('Value function')
    plt.colorbar(orientation='horizontal')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", type=str, required=True,
                        help="Grid Map file")
    parser.add_argument("-p", "--planner", type=str, default="PI",
                        help="Planner to use: [PI, VI], default: PI")

    args = parser.parse_args()
    main(args.map, args.planner)


import argparse

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
# from joblib import Parallel, delayed

plt.style.use('fivethirtyeight')
# import seaborn as sns

import numpy as np

from funzo.domains.gridworld import GridWorld, GridWorldMDP
from funzo.domains.gridworld import GReward, GRewardLFA, GTransition
from funzo.planners.dp import PolicyIteration, ValueIteration


def main(map_name, planner):
    print(map_name)
    gmap = np.loadtxt(map_name)
    w_expert = np.array([-0.01, -10.0, 1.0])

    world = GridWorld(gmap)
    # R = GReward(domain=world)
    R = GRewardLFA(domain=world, weights=w_expert, rmax=1.0)
    T = GTransition(domain=world, wind=0.1)

    g = GridWorldMDP(domain=world, reward=R, transition=T, discount=0.9)

    # ------------------------
    mdp_planner = PolicyIteration(verbose=0, max_iter=200,
                                  epsilon=1e-05, random_state=None)
    if planner == 'VI':
        mdp_planner = ValueIteration(verbose=2)

    # res = Parallel(n_jobs=4)(mdp_planner(g))
    res = mdp_planner(g)
    V = res['V']
    # print(V)
    # print(res['Q'])
    # print(res['pi'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = world.visualize(ax, policy=res['pi'])
    # plt.savefig('world.svg')

    # ------------------------

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='inferno', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.grid(False)
    plt.title('Value function')
    plt.colorbar(orientation='horizontal')
    # plt.savefig('world_value.svg')

    plt.figure()
    plt.plot(res['cR'])
    plt.xlabel('Iterations [t]')
    plt.ylabel('Accumulated rewards using $\pi_t$')
    plt.tight_layout()

    # rfunc = np.array([g.R(s, 4) for s in g.S])
    # plt.figure(figsize=(8, 8))
    # plt.imshow(rfunc.reshape(gmap.shape),
    #            interpolation='nearest', cmap='inferno', origin='lower',
    #            vmin=np.min(rfunc), vmax=np.max(rfunc))
    # plt.title('Reward function')
    # plt.grid(False)
    # plt.colorbar(orientation='horizontal')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", type=str, required=True,
                        help="Grid Map file")
    parser.add_argument("-p", "--planner", type=str, default="PI",
                        help="Planner to use: [PI, VI], default: PI")

    args = parser.parse_args()
    main(args.map, args.planner)

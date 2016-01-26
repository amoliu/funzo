
import argparse
from matplotlib import pyplot as plt

import numpy as np

from funzo.domains.gridworld import GridWorld, GRewardLFA
from funzo.planners.dp import policy_iteration

from funzo.irl.birl.map_birl import MAPBIRL
from funzo.irl.birl.base import GaussianRewardPrior


def main():
    gmap = np.loadtxt('maps/map_a.txt')
    rfunc = GRewardLFA(None, weights=[0.00, -0.1, 1.0])
    g = GridWorld(gmap, reward_function=rfunc, discount=0.7)

    # ------------------------
    plan = policy_iteration(g, verbose=1)
    policy = plan['pi']
    print(policy)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = g.visualize(ax, policy=policy)
    plt.show()

    demos = g.generate_trajectories(10, policy, random_state=None)
    # np.save('demos.npy', demos)
    # demos = np.load('demos.npy')
    # print(demos)

    # IRL
    r_prior = GaussianRewardPrior(sigma=0.75)
    irl_solver = MAPBIRL(mdp=g, prior=r_prior, demos=demos,
                         planner=policy_iteration, beta=0.8)
    r = irl_solver.run()

    g.reward.weights = r
    r_plan = policy_iteration(g)
    print(r_plan['pi'])
    print(r)
    V = r_plan['V']

    # ------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = g.visualize(ax, policy=r_plan['pi'])

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='viridis', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.title('Value function')
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main()

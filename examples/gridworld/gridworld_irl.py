
import argparse
from matplotlib import pyplot as plt

import numpy as np

from funzo.domains.gridworld import GridWorld
from funzo.planners.dp import policy_iteration

from funzo.irl.birl.map_birl import MAPBIRL
from funzo.irl.birl.base import GaussianRewardPrior


def main():
    gmap = np.loadtxt('maps/map_a.txt')
    g = GridWorld(gmap, discount=0.55)

    # ------------------------
    plan = policy_iteration(g, verbose=1)
    policy = plan['pi']
    print(policy)

    demos = g.generate_trajectories(5, policy, random_state=None)
    # np.save('demos.npy', demos)
    # demos = np.load('demos.npy')
    print(demos)

    # IRL
    r_prior = GaussianRewardPrior()
    irl_solver = MAPBIRL(mdp=g, prior=r_prior, demos=demos,
                         planner=policy_iteration, beta=0.7)
    r = irl_solver.run()

    g.reward.weights = r
    r_plan = policy_iteration(g)
    print(r_plan['pi'])
    print(r)
    print(r_plan['V'])

    # ------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = g.visualize(ax, policy=r_plan['pi'])

    plt.show()


if __name__ == '__main__':
    main()



import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np

from funzo.domains.puddleworld import PuddleWorld, PuddleWorldMDP
from funzo.domains.puddleworld import PuddleReward, PuddleRewardLFA
from funzo.domains.puddleworld import PWTransition
from funzo.planners import PolicyIteration


def main():

    with PuddleWorld(start=(0.5, 0.1), resolution=0.05) as world:
        # R = PuddleReward(rmax=1.0, step_reward=0.1)
        R = PuddleRewardLFA(weights=[1, -1], rmax=1.0)
        T = PWTransition()
        g = PuddleWorldMDP(reward=R, transition=T, discount=0.98)

        # ------------------------
        mdp_planner = PolicyIteration()
        res = mdp_planner.solve(g)
        V = res['V']
        print(V)
        print(res['pi'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = world.visualize(ax, policy=res['pi'])
    # plt.savefig('world.svg')

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(world.shape).T,  # interpolation='nearest',
               cmap='viridis', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.grid(False)
    plt.title('Value function')
    plt.colorbar(orientation='horizontal')
    # plt.savefig('world_value.svg')

    plt.show()


if __name__ == '__main__':
    main()

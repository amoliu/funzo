

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

import numpy as np

from funzo.domains.puddleworld import PuddleWorld, PuddleWorldMDP
from funzo.domains.puddleworld import PuddleReward
from funzo.domains.puddleworld import PWTransition
from funzo.planners.dp import PolicyIteration, ValueIteration


def main():
    world = PuddleWorld(start=(0.5, 0.5), resolution=0.1)
    R = PuddleReward(domain=world)
    T = PWTransition(domain=world)

    g = PuddleWorldMDP(domain=world, reward=R, transition=T, discount=0.9)

    # ------------------------
    mdp_planner = PolicyIteration(verbose=0)
    res = mdp_planner(g)
    V = res['V']
    print(V)
    # print(res['Q'])
    print(res['pi'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = world.visualize(ax, policy=res['pi'])
    # plt.savefig('world.svg')

    # ------------------------

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(world.shape),
               interpolation='nearest', cmap='inferno', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.grid(False)
    plt.title('Value function')
    plt.colorbar(orientation='horizontal')
    # plt.savefig('world_value.svg')

    plt.show()


if __name__ == '__main__':
    main()

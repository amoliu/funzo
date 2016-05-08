from __future__ import division

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')


from funzo.domains.chainworld import ChainWorld, ChainMDP
from funzo.domains.chainworld import ChainReward
from funzo.domains.chainworld import ChainTransition
from funzo.planners import PolicyIteration


def main():
    NUM_STATES = 10

    with ChainWorld(num_states=NUM_STATES) as world:
        R = ChainReward()
        T = ChainTransition()
        mdp = ChainMDP(R, T, discount=0.98)

        planner = PolicyIteration()
        plan = planner.solve(mdp)

        print(plan['pi'])

    fig = plt.figure(figsize=(12, 3))
    ax = fig.gca()
    ax = world.visualize(ax)
    ax = world.show_policy(ax, policy=plan['pi'])

    plt.figure(figsize=(8, 8))
    plt.plot(plan['V'])
    plt.title('Value function')

    plt.show()


if __name__ == '__main__':
    main()

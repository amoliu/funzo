

from matplotlib import pyplot as plt

import numpy as np

from pylfd.domains.gridworld import GridWorld, plot_values, plot_policy
from pylfd.utils.data_structures import ValueFunction, Policy


def main():
    gmap = [
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 2],
        # [0, 0, 1, 0]
    ]
    g = GridWorld(gmap)

    fig = plt.figure()
    ax = fig.gca()

    ax = g.visualize(ax)

    # ------------------------
    policy = Policy(g.S, g.A)
    # print(policy)

    V = ValueFunction(g.S)
    V[g.S[4]] = 10
    V[g.S[8]] = 6
    V[g.S[1]] = 2

    print(list(V.values()))

    fig = plt.figure()
    plot_values(V, fig.gca(), (4, 4))

    ##
    fig2 = plt.figure()
    plot_policy(policy, fig2.gca(), (4, 4))

    plt.show()


if __name__ == '__main__':
    main()

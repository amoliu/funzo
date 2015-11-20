

from matplotlib import pyplot as plt

import numpy as np

from pylfd.domains.gridworld import GridWorld
from pylfd.utils.data_structures import ValueFunction, Policy


def main():
    gmap = [
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 2],
        # [0, 0, 1, 0]
    ]
    # g = GridWorld(np.array(np.rot90(gmap)))
    g = GridWorld(gmap)
    # print(g._grid)

    # fig = plt.figure(figsize=(8, 8))
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
    # plt.plot(V.values())

    vmap = np.zeros(shape=(4, 4))
    for k, v in V.items():
        vmap[k.cell[0], k.cell[1]] = v

    plt.imshow(vmap, interpolation='nearest', cmap='viridis')
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main()

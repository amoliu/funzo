

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

    ##
    plt.figure()
    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
    U = np.cos(X)
    V = np.sin(Y)

    Q = plt.quiver(U, V)
    qk = plt.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
                       fontproperties={'weight': 'bold'})

    plt.show()


if __name__ == '__main__':
    main()

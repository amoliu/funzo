

from matplotlib import pyplot as plt

import numpy as np

from pylfd.domains.gridworld import GridWorld


def main():
    gmap = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 2, 0],
        # [0, 0, 1, 0]
    ]
    # g = GridWorld(np.array(np.rot90(gmap)))
    g = GridWorld(gmap)
    # print(g._grid)

    # fig = plt.figure(figsize=(8, 8))
    fig = plt.figure()
    ax = fig.gca()

    ax = g.visualize(ax)

    # fig = plt.figure()
    # plt.imshow(gmap, interpolation='nearest')

    plt.show()


if __name__ == '__main__':
    main()

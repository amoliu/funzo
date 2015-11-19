

from matplotlib import pyplot as plt

import numpy as np

from pylfd.domains.gridworld import GridWorld


def main():
    gmap = [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
        [0, 0, 1, 0]
    ]
    g = GridWorld(np.array(gmap).T)
    # print(g._grid)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    ax = g.visualize(ax)

    plt.show()


if __name__ == '__main__':
    main()

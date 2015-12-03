
import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt

import numpy as np

from funzo.domains.gridworld import GridWorld
from funzo.planners.exact import value_iteration, policy_iteration


def main(map_name):
    print(map_name)
    gmap = np.loadtxt(map_name)

    g = GridWorld(gmap, discount=0.6)

    # ------------------------
    # res = policy_iteration(g, verbose=2)
    res = value_iteration(g)
    V = res['V']
    print(V)
    # print(res['Q'])
    print(res['pi'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = g.visualize(ax, policy=res['pi'])

    # ------------------------

    plt.figure()
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='viridis', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    import sys
    main(sys.argv[1])

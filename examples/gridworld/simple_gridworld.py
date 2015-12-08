
import argparse
from matplotlib import pyplot as plt

import numpy as np

from funzo.domains.gridworld import GridWorld
from funzo.planners.exact import value_iteration, policy_iteration


def main(map_name, planner):
    print(map_name)
    gmap = np.loadtxt(map_name)

    g = GridWorld(gmap, discount=0.55)

    # ------------------------
    if planner == 'PI':
        res = policy_iteration(g, verbose=2)
    else:
        res = value_iteration(g)
    V = res['V']
    print(V)
    # print(res['Q'])
    print(res['pi'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = g.visualize(ax, policy=res['pi'])

    # ------------------------

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='viridis', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", type=str, required=True,
                        help="Grid Map file")
    parser.add_argument("-p", "--planner", type=str, default="PI",
                        help="Planner to use: [PI, VI], default: PI")

    args = parser.parse_args()
    main(args.map, args.planner)

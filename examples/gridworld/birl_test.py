
from __future__ import division

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import corner

import numpy as np

from funzo.domains.gridworld import GridWorld, GridWorldMDP
from funzo.domains.gridworld import GRewardLFA, GTransition
from funzo.planners.dp import PolicyIteration

from funzo.irl.birl import BIRL
from funzo.irl.birl import GaussianRewardPrior
from funzo.irl import PolicyLoss, RewardLoss


SEED = None


def main():
    gmap = np.loadtxt('maps/map_a.txt')
    w_expert = np.array([-0.01, -3.0, 1.0])
    # w_expert = np.array([-3.0, 1.0])
    w_expert /= (w_expert.max() - w_expert.min())

    world = GridWorld(gmap=gmap)
    RMAX = 1.0

    rfunc = GRewardLFA(domain=world, weights=w_expert, rmax=RMAX)
    T = GTransition(domain=world)
    g = GridWorldMDP(domain=world, reward=rfunc, transition=T, discount=0.9)

    # ------------------------
    planner = PolicyIteration(verbose=2)
    plan = planner(g)
    policy = plan['pi']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = world.visualize(ax, policy=policy)
    plt.show()

    # demos = np.load('demos.npy')
    demos = world.generate_trajectories(policy, num=100, random_state=SEED)

    # IRL
    r_prior = GaussianRewardPrior(dim=len(rfunc), mean=0.0, sigma=0.5)
    irl_solver = BIRL(prior=r_prior, delta=0.3, planner=planner, beta=0.7,
                      max_iter=1000, burn_ratio=0.3, inference='PW',
                      random_state=SEED)

    trace = irl_solver.solve(mdp=g, demos=demos)
    trace.save('pw_trace')
    r = trace['r_mean'][-1]
    # r = trace['r_map'][-1]

    g.reward.update_parameters(reward=r)
    r_plan = planner(g)
    print(r_plan['pi'])
    print('Found reward: {}'.format(r))
    V = r_plan['V']

    # compute the loss
    loss_func = RewardLoss(order=2)
    # loss_func = PolicyLoss(mdp=g, planner=planner, order=1)
    loss = [loss_func(w_expert, w_pi) for w_pi in trace['r']]
    loss_m = [loss_func(w_expert, w_pi) for w_pi in trace['r_mean']]

    # ------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = world.visualize(ax, policy=r_plan['pi'])

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='viridis', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.grid(False)
    plt.title('Value function')
    plt.colorbar()

    plt.figure(figsize=(8, 6))
    # plt.plot(loss)
    plt.plot(trace['step'], loss)
    plt.plot(trace['step'], loss_m)
    plt.ylabel('Loss function $\mathcal{L}_{\pi}$')
    plt.xlabel('Iteration')
    plt.tight_layout()

    # plt.figure()
    # plt.plot(trace['step'], trace['a_ratio'])
    # plt.ylabel('Acceptance probability')
    # plt.xlabel('Step')
    # plt.tight_layout()

    # if len(trace['sample']) > 100:
    #     corner.corner(trace['sample'])

    plt.show()


if __name__ == '__main__':
    main()

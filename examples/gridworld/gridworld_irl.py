
from __future__ import division

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import corner

import numpy as np

from funzo.domains.gridworld import GridWorld, GridWorldMDP
from funzo.domains.gridworld import GRewardLFA, GReward, GTransition
from funzo.planners.dp import PolicyIteration

from funzo.irl.birl.map_birl import MAPBIRL
from funzo.irl.birl.mcmc_birl import PolicyWalkBIRL
from funzo.irl.birl import GaussianRewardPrior, UniformRewardPrior
from funzo.irl import PolicyLoss, RewardLoss

from funzo.utils.diagnostics import plot_geweke_test
from funzo.utils.diagnostics import plot_sample_autocorrelations
from funzo.utils.diagnostics import plot_variable_histograms
from funzo.utils.diagnostics import plot_sample_traces

SEED = 42


def main():
    gmap = np.loadtxt('maps/map_a.txt')
    # w_expert = np.array([-0.001, -0.5, 1.0])
    w_expert = np.array([-0.01, -10.0, 1.0])
    w_expert /= (w_expert.max() - w_expert.min())

    world = GridWorld(gmap=gmap)
    # RMAX = 1.0/len(world.states)
    RMAX = 1.0

    print('RMAX', RMAX)

    # rfunc = GReward(domain=world, rmax=1.0/len(world.states))
    rfunc = GRewardLFA(domain=world, weights=w_expert,
                       rmax=RMAX)

    T = GTransition(domain=world)
    g = GridWorldMDP(domain=world, reward=rfunc, transition=T, discount=0.8)

    # w_expert = rfunc._R

    # ------------------------
    planner = PolicyIteration(verbose=2)
    plan = planner(g)
    policy = plan['pi']
    print(policy)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.gca()
    # ax = world.visualize(ax, policy=policy)
    # plt.show()

    demos = world.generate_trajectories(policy, num=20, random_state=SEED)
    # np.save('demos.npy', demos)
    # demos = np.load('demos.npy')
    # print(demos)

    # IRL
    # r_prior = GaussianRewardPrior(sigma=0.15)
    r_prior = UniformRewardPrior(loc=-RMAX, scale=2*RMAX)

    # irl_solver = MAPBIRL(mdp=g, prior=r_prior, demos=demos, planner=planner,
    #                      beta=0.6)
    irl_solver = PolicyWalkBIRL(mdp=g, prior=r_prior, demos=demos,
                                delta=0.3, planner=planner, beta=0.8,
                                max_iter=500, cooling=True, burn_ratio=0.2)

    trace = irl_solver.run(random_state=SEED)
    trace.save('pw_trace')
    r = trace['r_mean'][-1]
    # r = trace['r_map'][-1]

    g.reward.update_parameters(reward=r)
    r_plan = planner(g)
    print(r_plan['pi'])
    print('Found reward: {}'.format(r))
    V = r_plan['V']

    # compute the loss
    # loss_func = PolicyLoss(mdp=g, planner=planner, order=1)
    loss_func = RewardLoss(order=2)
    # loss = [loss_func(w_expert, w_pi) for w_pi in trace['r']]
    loss = [loss_func(w_expert, w_pi) for w_pi in trace['r_mean']]

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
    # plt.plot(data['iter'], loss)
    plt.plot(trace['step'], loss)
    plt.ylabel('Loss function $\mathcal{L}_{\pi}$')
    plt.xlabel('Iteration')
    plt.tight_layout()

    # figure = corner.corner(trace['r'])
    figure = corner.corner(trace['sample'])

    # plot_geweke_test(trace['r'])
    # plot_sample_autocorrelations(np.array(trace['r']), thin=5)
    # plot_variable_histograms(np.array(trace['r']))

    plt.show()


if __name__ == '__main__':
    main()

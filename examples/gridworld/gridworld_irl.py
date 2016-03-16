

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np

from funzo.domains.gridworld import GridWorld, GridWorldMDP
from funzo.domains.gridworld import GRewardLFA, GReward, GTransition
from funzo.planners.dp import PolicyIteration

from funzo.irl.birl.map_birl import MAPBIRL
from funzo.irl.birl.mcmc_birl import PolicyWalkBIRL
from funzo.irl.birl import GaussianRewardPrior
from funzo.irl import PolicyLoss, RewardLoss

SEED = None


def main():
    gmap = np.loadtxt('maps/map_a.txt')
    # w_expert = np.array([0.001, -0.1, 1.0])
    # w_expert = np.array([-0.001, -0.1, 1.0])
    w_expert = np.array([0.01, -0.5, 1.0])

    world = GridWorld(gmap=gmap)
    # rfunc = GReward(domain=world)
    rfunc = GRewardLFA(domain=world, weights=w_expert)
    T = GTransition(domain=world)
    g = GridWorldMDP(domain=world, reward=rfunc, transition=T, discount=0.7)

    # w_expert = rfunc._R

    # ------------------------
    planner = PolicyIteration(verbose=2)
    plan = planner(g)
    policy = plan['pi']
    print(policy)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.gca()
    # ax = g.visualize(ax, policy=policy)
    # plt.show()

    demos = world.generate_trajectories(policy, num=50, random_state=None)
    # np.save('demos.npy', demos)
    # demos = np.load('demos.npy')
    # print(demos)

    # IRL
    r_prior = GaussianRewardPrior(sigma=0.15)
    # irl_solver = MAPBIRL(mdp=g, prior=r_prior, demos=demos, planner=planner,
    #                      beta=0.6)
    irl_solver = PolicyWalkBIRL(mdp=g, prior=r_prior, demos=demos, delta=0.3,
                                planner=planner, beta=0.6, max_iter=50)
    # r, data = irl_solver.run(random_state=None)
    trace, r = irl_solver.run(random_state=None)

    g.reward.weights = r
    r_plan = planner(g)
    print(r_plan['pi'])
    print(r)
    V = r_plan['V']

    # compute the loss
    # loss_func = PolicyLoss(mdp=g, planner=planner, order=1)
    loss_func = RewardLoss(order=2)
    # loss = [loss_func(w_expert, w_pi) for w_pi in data['rewards']]
    loss = [loss_func(w_expert, w_pi) for w_pi in trace['r']]

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

    plt.show()


if __name__ == '__main__':
    main()

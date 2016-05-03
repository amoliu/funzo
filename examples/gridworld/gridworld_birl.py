
from __future__ import division

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np

from funzo.domains.gridworld import GridWorld, GridWorldMDP
from funzo.domains.gridworld import GRewardLFA, GTransition, GReward
from funzo.planners.dp import PolicyIteration

from funzo.irl.birl import PolicyWalkBIRL, MAPBIRL
from funzo.irl.birl import GaussianRewardPrior
from funzo.irl import PolicyLoss, RewardLoss


SEED = None


def main():
    gmap = np.loadtxt('maps/map_a.txt')

    w_expert = np.array([-0.01, -3.0, 1.0])
    w_expert /= (w_expert.max() - w_expert.min())

    with GridWorld(gmap=gmap) as world:
        rfunc = GRewardLFA(weights=w_expert, rmax=1.0)
        # rfunc = GReward()
        T = GTransition()
        g = GridWorldMDP(reward=rfunc, transition=T, discount=0.99)

        # ------------------------
        planner = PolicyIteration(random_state=SEED)
        plan = planner.solve(g)
        policy = plan['pi']

        demos = world.generate_trajectories(policy, num=150, random_state=SEED)

        # IRL
        r_prior = GaussianRewardPrior(dim=len(rfunc), mean=0.0, sigma=0.35)
        irl_solver = PolicyWalkBIRL(prior=r_prior, delta=0.2, planner=planner,
                                    beta=0.8, max_iter=1000, burn=0.3,
                                    random_state=SEED)

        trace = irl_solver.solve(demos=demos, mdp=g)
        trace.save('pw_trace')
        r = trace['r_mean'][-1]

        g.reward.update_parameters(reward=r)
        r_plan = planner.solve(g)
        print('Found reward: {}'.format(r))
        V = r_plan['V']

        # w_expert = rfunc._R

    # compute the loss
    L = RewardLoss(order=2)
    # L = PolicyLoss(mdp=g, planner=planner, order=2)
    loss = [L.evaluate(w_expert, w_pi) for w_pi in trace['r']]
    loss_m = [L.evaluate(w_expert, w_pi) for w_pi in trace['r_mean']]

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
    plt.plot(trace['step'], loss)
    plt.plot(trace['step'], loss_m)
    plt.ylabel('Loss function $\mathcal{L}_{\pi}$')
    plt.xlabel('Iteration')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()

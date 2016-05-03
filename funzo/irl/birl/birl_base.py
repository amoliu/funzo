"""
BIRL algorithms

"""
from __future__ import division


from scipy.misc import logsumexp

from ..irl_base import IRLSolver
from ...utils.validation import check_random_state


class BIRLBase(IRLSolver):
    """ Bayesian IRL algorithm

    BIRL algorithms seek to find a reward function underlying a set of
    expert demonstrations by computing the reward posterior distribution
    :math:`p(r | D)`.

    These algorithms typically can return a single reward estimate by
    computing the various quantities e.g. mean

    Parameters
    ----------
    prior : :class:`RewardPriorBase` object
        Reward prior callable object
    beta : float, optional (default=0.7)
        Expert optimality parameter
    planner : a callable, optional (default=None)
        A planner for MDP e.g. policy iteration as a callable
    random_state : :class:`numpy.RandomState`, optional (default: None)
        Random number generation seed


    Attributes
    ----------
    _prior : :class:`RewardPriorBase` instance
        Reference to the reward prior callable object
    _beta : float
        Expert optimality parameter
    _rng : :class:`numpy.RandomState`
        Random number generator

    """

    def __init__(self, prior, beta=0.7, planner=None, random_state=None):
        super(BIRLBase, self).__init__(planner)
        self._prior = prior
        self._beta = beta
        self._rng = check_random_state(random_state)

    def initialize_reward(self):
        """ Initialize a reward vector using the prior distribution """
        r = self._prior.sample()
        return r

    def log_posterior(self, r, demos, mdp, plan_r):
        """ Reward posterior distribution (unnormalized)

        .. math::

            \log p(r | D) = \log p(D | r) + \log p(r)

        """
        llk = self.log_likelihood(plan_r['Q'], demos, mdp)
        lp = self.log_prior(r)
        return llk + lp

    def log_likelihood(self, Q_r, demos, mdp):
        """ Evaluate the log likelihood of the demonstrations w.r.t reward

        .. math::

            \log p(D | r) = \sum_{d \in D} \sum_{(s,a) \in d}
            \left((\\beta Q(s,a;r)) - \log \sum_{b \in A} (\\beta Q(s,b;r))
            \\right)

        where :math:`d` are trajectories or sets of state-action pairs.

        """
        llk = 0.0
        M = len(demos)
        for traj in demos:
            if len(traj) > 0:
                H = len(traj)
                alpha_H = 0.0
                beta_H = 0.0
                for (s, a) in traj:
                    alpha_H += self._beta * Q_r[a, s]
                    beta_Hs = [self._beta * Q_r[b, s] for b in mdp.A]
                    beta_H += logsumexp(beta_Hs)

                llk += (alpha_H - beta_H) / float(H)
        llk /= float(M)

        return llk

    def log_prior(self, r):
        """ Compute log prior probability

        Scales with the dimension of the reward function.

        .. math::

            \log p(r) = \sum_i p(r(s_i, a_i))

        """
        return self._prior.log_p(r)

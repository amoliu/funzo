"""
BIRL algorithms with MCMC based inference
"""

from __future__ import division, absolute_import

import six

import numpy as np

from abc import abstractmethod, ABCMeta
from tqdm import tqdm
from copy import deepcopy
from six.moves import range

from .birl_base import BIRLBase
from ...base import Model
from ...utils.validation import check_random_state
from ...utils.data_structures import Trace


__all__ = ['PolicyWalkBIRL', 'PolicyWalkProposal']


class PolicyWalkBIRL(BIRLBase):
    """ BIRL using PolicyWalk algorithm for inference """
    def __init__(self, prior, beta=0.7, delta=0.2, max_iter=100, burn=0.27,
                 planner=None, random_state=None):
        super(PolicyWalkBIRL, self).__init__(prior, beta,
                                             planner, random_state)

        if 0 >= max_iter > np.inf:
            raise ValueError('No. of iterations must be in (0, inf)')
        self._max_iter = max_iter

        if 0.0 > burn >= 1.0:
            raise ValueError('burn ratio must be in [0, 1)')
        self._burn = int(self._max_iter * burn / 100.0)

        if 0.0 >= delta > 1.0:
            raise ValueError('Reward steps (delta) must be in (0, 1)')
        self._delta = delta

    def solve(self, demos, mdp=None):
        """ Solve the BIRL problem using PolicyWalk """
        if mdp is None:
            raise ValueError('BIRL requires an MDP model')

        v = ['step', 'r', 'r_mean', 'sample', 'a_ratio']
        trace = Trace(v, save_interval=self._max_iter // 2)

        proposal = PolicyWalkProposal(dim=len(mdp.reward), delta=self._delta)

        r = self.initialize_reward()
        plan_r = self.solve_mdp(mdp, r)

        r_mean = np.array(r)
        for step in tqdm(range(1, self._max_iter + 1), desc='PolicyWalk'):
            r_new = proposal.step(r)
            plan_r_new = self.solve_mdp(mdp, r_new, plan_r['V'], plan_r['pi'])
            p_accept = self._acceptance_ratio(mdp, demos, r, r_new,
                                              plan_r, plan_r_new)
            if self._rng.uniform() < min([1.0, p_accept]):
                r = np.array(r_new)
                plan_r = deepcopy(plan_r_new)

            # if step > self._burn:
            r_mean = self._iterative_mean(r_mean, r, step)
            trace.record(step=step, r=r, r_mean=r_mean, sample=r_new,
                         a_ratio=p_accept)

        return trace

    def _acceptance_ratio(self, mdp, demos, r, r_new, plan_r, plan_r_new):
        """ Compute PolicyWalk acceptance ratio """
        lp_r = self.log_posterior(r, demos, mdp, plan_r)
        lp_r_new = self.log_posterior(r_new, demos, mdp, plan_r_new)
        return lp_r_new / lp_r

    def _iterative_mean(self, r_mean, r_new, step):
        """ Compute the iterative mean of the reward """
        r_mean = [((step - 1) / float(step)) *
                  r_m + 1.0 / step * r for r_m, r in zip(r_mean, r_new)]
        return np.array(r_mean)


########################################################################


class MCMCProposalBase(six.with_metaclass(ABCMeta, Model)):
    """ Proposal for MCMC sampling """
    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def step(self, location):
        """ Take a single MCMC chain step """
        raise NotImplementedError('abstract')


class PolicyWalkProposal(MCMCProposalBase):
    """ PolicyWalk MCMC proposal

    Sampling in the hypercube with limits [-rmax, rmax]

    """
    def __init__(self, dim, delta, rmax=1.0, random_state=None):
        super(PolicyWalkProposal, self).__init__(dim)
        self.delta = delta
        self.rmax = rmax
        self.rng = check_random_state(random_state)

    def step(self, location):
        """ Take a single MCMC chain step

        PolicyWalk takes steps in the grid defined by,

        .. math::

            \mathbb{R}^{|R|} / \delta

        where :math:`|R|` is the dimension of the reward space, which can get
        very large. In this case a lot of samples are needed before the MCMC
        chain converges.

        """
        sample = np.array(location)
        d = self.rng.choice([-self.delta, self.delta])
        i = self.rng.randint(self.dim)
        if -self.rmax <= sample[i] + d <= self.rmax:
            sample[i] += d
        return sample

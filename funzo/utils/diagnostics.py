

import numpy as np

from matplotlib import pyplot as plt


def geweke_test(trace, intervals=10, length=100, first=10):
    """ A formal test for MCMC chain convergence """
    nsl = length
    # first = 0.1*len(trace)

    z = np.empty(intervals)
    for k in np.arange(0, intervals):
        # beg of each sub samples
        bega = first+k*length
        begb = len(trace)/2 + k*length

        sub_trace_a = trace[bega:bega+nsl]
        sub_trace_b = trace[begb:begb+nsl]

        theta_a = np.mean(sub_trace_a)
        theta_b = np.mean(sub_trace_b)
        var_a = np.var(sub_trace_a)
        var_b = np.var(sub_trace_b)

        z[k] = (theta_a - theta_b) / np.sqrt(var_a + var_b)

    return z


def plot_geweke_test(trace, intervals=10, length=100, first=10, **metadata):
    """
    Plot the the nature of the Geweke test for MCMC convergence
    """
    z = geweke_test(trace, intervals, length, first)

    # get any metadata
    # algorithm = metadata['algorithm']
    # reward_type = metadata['reward_type']

    plt.figure(figsize=(8, 6))
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    ax.plot([2]*len(z), 'r-')
    ax.plot(z)
    ax.plot([-2]*len(z), 'r-')
    ax.axhspan(-2, 2, facecolor='g', alpha=0.15, label='Convergence region')
    # ax.axhspan(-2, 2, facecolor='g', fill=False, hatch='\\', alpha=0.15,
    #             label='Convergence region')
    ax.set_ylim([-3, 3])
    ax.set_ylabel('$z$')
    ax.legend(loc='best')
    ax.set_title('Geweke Test')

    return ax


def plot_reward_samples(reward_traces, **metadata):
    """
    Plot the MCMC walk on the space of rewards for lower dimensional
    reward functions (2 dim)
    """
    # get any metadata
    m_rejects = reward_traces[reward_traces[:, 0] == 0]
    m_accepted = reward_traces[reward_traces[:, 0] == 1]
    mr = np.mean(m_accepted[:, 1:], axis=0)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    ax.plot(m_rejects[:, 1], m_rejects[:, 2], color='k',
            marker='*', ls='-', lw=1, markersize=8, alpha=0.3)
    ax.plot(m_accepted[:, 1], m_accepted[:, 2], color='b',
            marker='o', ls='-', lw=1, markersize=8, alpha=0.5)

    ax.plot(mr[0], mr[1], color='r', marker='8', ls='-',
            lw=1, markersize=15, alpha=0.5)

    plt.axis('equal')
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_xlabel('Feature $\phi_1(s)$')
    ax.set_ylabel('Feature $\phi_2(s)$')
    ax.set_title('MCMC chain walk')
    plt.tight_layout()

    return ax


def plot_sample_traces(trace, burnin=0.27, **metadata):
    """ Plot MCMC traces """
    algorithm = metadata['algorithm']
    reward_type = metadata['reward_type']
    burnin = float(burnin)
    burn = int(trace.shape[0] * (burnin / 100.0))

    plt.figure(figsize=(12, 5))
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    for f in xrange(1, trace.shape[1], 1):
        ax.plot(trace[:, f], ls='-', alpha=0.6, lw=1.5,
                label='$\phi_{0}(s)$'.format(f))

    ax.axvspan(0, burn, facecolor='r', alpha=0.15, label='Burn in')
    # ax.set_ylim([-1.1, 1.1])
    ax.legend(loc='best')
    ax.set_xlabel('Iteration')
    ax.set_title('MCMC samples trace: ' + algorithm + ' - ' + reward_type)
    plt.tight_layout()

    return ax


def plot_sample_autocorrelations(trace, burnin=0.27, **metadata):
    """ Plot Autocorrelations """
    thin = metadata['thin']
    burnin = float(burnin)
    burn = int(trace.shape[0] * (burnin / 100.0))
    no_variables = trace.shape[1]

    fig, axes = plt.subplots(no_variables, 1, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for i in range(no_variables):
        sig = trace[burn:, i]
        mlag = len(sig) // thin

        axes[i].acorr(sig[0:len(sig):thin], normed=True, usevlines=True,
                      maxlags=mlag, alpha=0.6, lw=1.5)
        axes[i].set_xlim([0, mlag])
        axes[i].set_title('$\phi_{0}(s)$'.format(i + 1))

    fig.suptitle('MCMC samples trace ACF')


def plot_variable_histograms(trace, burnin, **metadata):
    """ Plot the different histograms for each variable """
    algorithm = metadata['algorithm']
    reward_type = metadata['reward_type']
    burnin = float(burnin)
    burn = int(trace.shape[0] * (burnin / 100.0))
    no_variables = trace.shape[1] - 1

    fig, axes = plt.subplots(1, no_variables, figsize=(12, 6))
    axes = axes.ravel()
    for i in range(no_variables):
        axes[i].hist(trace[burn:, i+1], histtype="step", lw=1.5,
                     bins=20, alpha=0.9)
        axes[i].set_title('$\phi_{0}(s)$'.format(i + 1))

    fig.suptitle('MCMC samples histograms: ' + algorithm + ' - ' + reward_type)

    return axes

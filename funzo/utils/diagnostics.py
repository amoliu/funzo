
# Some of these functions are borrowed from pymc3

import numpy as np


def autocorr(x, lag=1):
    """ Sample autocorrelation at specified lag.

    The autocorrelation is the correlation of x_i with x_{i+lag}.

    """
    S = autocov(x, lag)
    return S[0, 1]/np.sqrt(np.prod(np.diag(S)))


def autocov(x, lag=1):
    """ Sample autocovariance at specified lag.

    The autocovariance is a 2x2 matrix with the variances of x[:-lag] and
    x[lag:] in the diagonal and the autocovariance on the off-diagonal.

    """
    x = np.asarray(x)

    if not lag:
        return 1
    if lag < 0:
        raise ValueError("Autocovariance lag must be a positive integer")
    return np.cov(x[:-lag], x[lag:], bias=1)


def geweke(x, first=.1, last=.5, intervals=20):
    """Return z-scores for convergence diagnostics.

    Compare the mean of the first \% of series with the mean of the last \% of
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.

    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each
      interval and score the Geweke score on the interval.

    Notes
    -----
    The Geweke score on some series x is computed by:

        .. math::
            \\frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    where :math:`E` stands for the mean, :math:`V` the variance,
    :math:`x_s` a section at the start of the series and
    :math:`x_e` a section at the end of the series.

    This implementation is borrowed from pymc3

    References
    ----------
    Geweke (1992)

    """

    if np.ndim(x) > 1:
        return [geweke(y, first, last, intervals) for y in np.transpose(x)]

    # Filter out invalid intervals
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first,
             last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Calculate starting indices
    sindices = np.arange(0, end // 2, step=int((end / 2) / (intervals - 1)))

    # Loop over start indices
    for start in sindices:

        # Calculate slices
        first_slice = x[start: start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(first_slice.std() ** 2 + last_slice.std() ** 2)

        zscores.append([start, z])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)

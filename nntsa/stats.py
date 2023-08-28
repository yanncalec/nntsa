"""
Statistics-related methods.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from . import tsa


def normalize(df:pd.DataFrame, idx:list=None):
    """Normalize a dataframe by removing the mean and rescaling the standard deviation.

    Args
    ----
    df:
        input dataframe, with the first dimension as time index.
    idx:
        index used for computing the statistics, a list. By default use all indexes.

    Returns
    -------
    dn:
        normalizaed dataframe.
    μ, σ:
        mean and std of the original dataframe.
    """
    if idx is None:
        μ = df.mean(axis=0)
        σ = df.std(axis=0)
    else:
        μ = df.loc[idx].mean(axis=0)
        σ = df.loc[idx].std(axis=0)
    dn = df - μ
    idx = σ > 1e-8
    dn.loc[:,idx] /= σ.loc[idx]
    return dn, μ, σ


def fit_cov_estimator(df:pd.DataFrame, *, estim='emp', **kwargs):
    """Fit a covariance estimator on a dataframe.

    Args
    ----
    df:
        input dataframe.
    estim:
        method for the covariance estimator, 'emp' or 'mcd'
    kwargs:
        keyword arguments for the covariance estimator, see :meth:`MinCovDet()` and :meth:`EmpiricalCovariance` of the module :mod:`sklearn.covariance`.
    """
    if estim == 'mcd':
        # fit a MCD robust estimator to data
        estimator = MinCovDet(**kwargs).fit(df)
    else:
        estimator = EmpiricalCovariance(**kwargs).fit(df)
    return estimator


def robust_cov(X:np.ndarray, k:int, N=1000):
    """Sample covariance matrix by bagging.

    Args
    ----
    X:
        2d array of shape (feature, sample)
    k:
        size of random subset.
    N:
        number of draws.

    Returns
    -------
    C, S: tuple
        estimated covariance matrix and standard deviation.
    """
    foo = []

    for _ in range(N):
        idx = np.random.permutation(range(X.shape[1]))[:k]
        foo.append(np.cov(X[:, idx]))

    C = np.mean(foo, axis=0)
    S = np.std(foo, axis=0)
    return (C+C.T)/2, S


def cross_corr(x:pd.Series, y:pd.Series, **kwargs):
    """Cross-correlation between two time series.
    """
    cov2corr = lambda c: c[0,1] / np.sqrt(c[0,0]*c[1,1])
    estimator = fit_cov_estimator(pd.DataFrame([x,y]).T, **kwargs)
    return cov2corr(estimator.covariance_)


def rel_error(yo:pd.Series, yp:pd.Series):
    """Relative error between two time series.

    The relative error of a prediction vector :math:`y'` from an observation vector :math:`y` is defined as

    .. math::
        \\text{rel.error} = \\frac{\|y-y'\|}{\|y-\\mu_y\|}

    with :math:`\mu_y` being the mean of the observation.

    Args
    ----
    yo:
        vector of observation
    yp:
        vector of prediction
    """
    return np.linalg.norm(yp-yo) / np.linalg.norm(yo-np.mean(yo))


def robust_r2(yo:pd.Series, yp:pd.Series, *, nans='drop', **kwargs):
    """Robust R2 score between two time series.

    Args
    ----
    yo, yp:
        two input time series with the same length.
    nans:
        {'drop', 'fill'}, to drop or to remove nans.
    kwargs:
        keyword parameters to :meth:`technip.stats.fit_cov_estimator`.

    Returns
    -------
    r2, s2, corr, rerr: tuple
        R2 score, S2 score, cross-correlation and relative error.
    """
    yobs, yprd = pd.Series(yo), pd.Series(yp)
    assert np.all(yobs.index == yprd.index)

    if nans == 'drop':
        yobs, yprd = tsa.drop_nan_rows(yobs, yprd)
    else:
        yobs = yobs.interpolate().bfill().ffill()
        yprd = yprd.interpolate().bfill().ffill()

    # corr = np.corrcoef(yprd, yobs)[0,1]
    corr = cross_corr(yobs, yprd, **kwargs)
    rerr = rel_error(yobs, yprd)
    s2 = (corr/rerr)**2
    r2 = s2/(1+s2)

    return r2, s2, corr, rerr


def r2score(Yo:pd.DataFrame, Yp:pd.DataFrame, *, robust=False, **kwargs):
    """R2 score between two dataframes.

    Args
    ----
    Yo, Yp:
        observation and prediction, pandas DataFrame, must have the same column names.
    robust:
        compute the robust version R2 score, False by default.
    estim:
        'emp' or 'mcd' for covariance estimator, see `fit_cov_estimator()`.

    Note
    ----
    This function by default computes R2 using its original definition:

    .. math::
        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    To obtain the robust R2 score (always bounded in (0,1)) pass the keyword argument `estim`.
    """

    # Force conversion
    Yo = pd.DataFrame(Yo)
    Yp = pd.DataFrame(Yp)[Yo.columns]
    # assert Yo.columns == Yp.columns

    if robust:
        # robust R2 score, but much slower
        R2s = pd.Series({n:robust_r2(Yo[n], Yp[n], **kwargs)[0] for n in Yo.columns})
    else:
        R2s = 1 - ((Yp-Yo)**2).mean(axis=0) / Yo.var(axis=0)
    return R2s.to_frame(name='R2_score').T


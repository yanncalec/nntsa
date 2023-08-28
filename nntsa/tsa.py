"""
Methods for Time Series Analysis (TSA).
"""

import numpy as np
import pandas as pd


def drop_nan_rows(X:pd.DataFrame, *args, **kwargs) -> list:
    """Drop rows containing nans simultaneously in multiple dataframes.

    Args
    ----
    X:
        input dataframe, with the first dimension as time index.
    args:
        other inputs, if given must have the same length as X.
    kwargs:
        keyword arguments to :meth:`dropna`.

    Return
    ------
    A list containing dataframes without nans.
    """

    df = pd.DataFrame(X)
    foo = [df.values]
    dims = [df.shape[1]]

    for Y in args:
        df = pd.DataFrame(Y)
        foo.append(df.values)
        dims.append(dims[-1]+df.shape[1])

    foo = pd.DataFrame(np.hstack(foo), index=X.index).dropna(axis=0, **kwargs)
    ars = np.split(foo.values, dims[:-1], axis=1)

    # Xd = [pd.DataFrame(ars[0], columns=X.columns, index=foo.index)]
    # for n, a in enumerate(ars[1:]):
    #     Xd.append(pd.DataFrame(a, columns=args[n].columns, index=foo.index))

    if type(X) is pd.Series:
        Xd = [pd.Series(np.squeeze(ars[0]), name=X.name, index=foo.index)]
    elif type(X) is pd.DataFrame:
        Xd = [pd.DataFrame(ars[0], columns=X.columns, index=foo.index)]
    else:
        Xd = [pd.DataFrame(ars[0], index=X.index)]

    for n, a in enumerate(ars[1:]):
        Y = args[n]
        if type(Y) is pd.Series:
            xd = pd.Series(np.squeeze(a), name=Y.name, index=foo.index)
        elif type(Y) is pd.DataFrame:
            xd = pd.DataFrame(a, columns=Y.columns, index=foo.index)
        else:
            xd = pd.DataFrame(a, index=Y.index)
        Xd.append(xd)

    return Xd


def count_nan_along_axis(df:pd.DataFrame, axis=-1) -> int:
    """Count the number of subarray containing nans along given axis.
    """
    if df.ndim>1:
        return (df.isna().sum(axis=axis)>0).values
    else:
        return (df.isna()).values


def safe_dot(W:pd.DataFrame, x:pd.DataFrame):
    """NaN safe matrix-vector product.
    """
    x1 = np.dot(W, x.fillna(0))

    if x.ndim > 1:
        return x1.set_index(x.index)
    else:
        return pd.Series(x1, index=x.index, name=x.name)

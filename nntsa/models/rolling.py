"""
Rolling window estimator framework for linear models.
"""

# from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd

# import sklearn
# from sklearn import linear_model
# from .. import tsa

import logging
logger = logging.getLogger(__name__)


class RollingWindow:
    """Rolling window estimator.
    """
    def __init__(self, data_in, data_out, order, basename:str, wsize:int, pstep:int, *, train_idx:list=None):
        if isinstance(order, tuple):
            _order = order
            self._type = 'dynamic'
        elif isinstance(order, int):
            _order = (order, 0)
            self._type = 'static'

        self.data_in = data_in
        self.data_out = data_out
        self.basename = basename
        self.order = _order
        self.wsize = wsize
        self.pstep = pstep
        self.train_idx = train_idx

    @property
    def name(self):
        return f'{self.basename}{self.order}'

    @staticmethod
    def _fit_predict(data_in:pd.DataFrame, data_out:pd.DataFrame, order, basename:str, wsize:int, pstep:int, train_idx:list=None, *, mode='scalar', **kwargs):
        from . import get_stationary_model
        assert len(data_in) == len(data_out)

        # rolling window fit
        res = []
        for t in range(0,len(data_in), pstep):
            t0 = max(0,t-wsize)
            # time index of training data
            window_start, window_end = data_in.index[t0], data_in.index[t]
            logger.info(f'{window_start}, {window_end}')
            if train_idx is None:
                window_idx = data_in.index[t0:t]
            else:
                window_idx = train_idx[(window_start<=train_idx) * (train_idx<window_end)]

            if mode == 'scalar':
                try:
                    prd = {}
                    for output, datao in data_out.iteritems():
                        model = get_stationary_model(data_in, datao, order, basename, train_idx=window_idx, **kwargs).fit()
                        prd[output] = model.predict(insample=True).iloc[:,0]
                    res.append((t,pd.DataFrame(prd)))
                except Exception as msg:
                    logger.error(msg)
            else:
                try:
                    model = get_stationary_model(data_in, data_out, order, basename, train_idx=window_idx, **kwargs).fit()
                    prd = model.predict(insample=True)
                    res.append((t,prd))
                except Exception as msg:
            #         res.append((t,None))
                    logger.error(msg)

        # prediction
        prd = data_out * np.nan
        for t,val in res:
            t1 = t+1
            try:
                prd.iloc[t1:t1+pstep] = val.iloc[t1:t1+pstep]
            except Exception as msg:
                logger.error(msg)

        return prd, res

    def fit_predict(self, *, mode='scalar', **kwargs):
        return RollingWindow._fit_predict(self.data_in ,self.data_out, self.order, self.basename, self.wsize, self.pstep, self.train_idx, mode=mode, **kwargs)


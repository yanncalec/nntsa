"""
Collection of VARMAX (Vectorial Auto-Regressive Moving Average with eXogeneous inputs) models.

Under development.
"""

import numpy as np
from .varx import VARX
from .. import tsa

import statsmodels
# import statsmodels.base.model.LikelihoodModel
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.tsa.vector_ar import var_model


class VARMAX(VARX):
    """Vectorial Auto-Regressive Moving-Average model with eXogeneous inputs (VARX).
    """
    @property
    def order_ma(self):
        return self._order[2]

    @property
    def name(self):
        # name of the model
        foo = 'ARMAX'
        if np.squeeze(self.data_out.values).ndim > 1:
            # prefix for the vectorial case
            foo = 'V'+foo
        return foo+ +f'{self._order}-'+self.regmodel_name

    @property
    def coef_ma(self):
        """Fitted coefficients of the innovations (moving average).
        """
        try:
            return pd.DataFrame(self._coef_noise, columns=self.data_out.columns, index=range(self._order[2]))
        except:
            return None


class Model_statsmodels(VARMAX):
    """Virtual class based on the `statsmodels` package for parameters estimation.
    """

    def __init__(self, *args, train_period=None):
        """
        Args
        ----
        train_period:
            a tuple (start, end) defining the index range of training data.
        """
        super().__init__(*args)

        self.data_in_stack = VARMAX.stack(self.data_in, self._order[0])
        self.data_out_stack = VARMAX.stack(self.data_out, self._order[1], 1)
        # self.data_stack = self.data_in_stack.join(self.data_out_stack)

        if train_period is not None:
            train_start, train_end = train_period
            self._Xtrn, self._Ytrn = tsa.drop_nan_rows(self.data_in_stack.loc[train_start:train_end], self.data_out.loc[train_start:train_end])
        else:
            self._Xtrn, self._Ytrn = tsa.drop_nan_rows(self.data_in_stack, self.data_out)

    def predict_insample(self, data_in=None, data_out=None):
        if data_in is None:
            data_in_stack = self.data_in_stack.fillna(0)
        else:
            assert data_in.shape[1] == self.data_in.shape[1]
            data_in_stack = VARMAX.stack(data_in, self._order[0]).fillna(0)

        if data_out is None:
            data_out_stack = self.data_out_stack.fillna(0)
        else:
            assert data_out.shape[1] == self.data_out.shape[1]
            data_out_stack = VARMAX.stack(data_out, self._order[1], 1).fillna(0)

        assert np.all(data_in_stack.index == data_out_stack.index)
        data_stack = data_in_stack.join(data_out_stack)

        if self._coef_in.ndim == 2 or self._coef_out.ndim == 2:
            cm = np.vstack([self._coef_in, self._coef_out])
        else:
            cm = np.hstack([self._coef_in, self._coef_out])

        goo = np.dot(data_stack, cm) + np.atleast_1d(self._intercept)
        return pd.DataFrame(goo, index=data_in_stack.index, columns=self.data_out.columns)


class VARX_statsmodels(Model_statsmodels):
    """VARX based on `statsmodels`.

    Due to limitation of `statsmodels`, this class can only handle the vectorial case,  i.e. the number of targets must be > 1.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.model = var_model.VAR(self._Ytrn, self._Xtrn)

    @property
    def name(self):
        return f'VARX{(self.order_x, self.order_ar)}-sm'

    def _fit(self):
        self._model_fit = self.model.fit(self._order[1])
        # post-processing
        dim = self.data_in_stack.shape[1]
        self._intercept, self._coef_in, self._coef_out = np.split(self._model_fit.params.values, [1,1+dim], axis=0)


class ARMAX_statsmodels(Model_statsmodels):
    """ARMAX based on `statsmodels`.

    This class can only handle the scalar case, i.e. the number of target must be = 1.
    """
    def __init__(self, *args, train_period=None, trend='c', **kwargs):
        """
        Args
        ----
        order:
            a tuple (exogeneous lag, auto-regression lag, moving-average lag)
        """
        super().__init__(*args, train_period=train_period)
        self.model = sarimax.SARIMAX(self._Ytrn, self._Xtrn, order=(self._order[1], 0, self._order[2]), trend=trend, **kwargs)

    @property
    def name(self):
        return f'ARMAX{(self.order_x, self.order_ar, self.order_ma)}-sm'

    def _fit(self, **kwargs):
        self._model_fit = self.model.fit(**kwargs)
        self._model_ext = self._model_fit.extend(self.data_out.values, exog=self.data_in_stack.fillna(0).values)
        # post-processing
        # xdim = self.data_in_stack.shape[1]
        # self._intercept, self._coef_in, self._coef_out, self._coef_noise, self._variance = np.split(self._model_fit.params.values, \
        #     [1,1+xdim, 1+xdim+self._order[1], 1+xdim+self._order[1]+self._order[2]], axis=-1)
        self._intercept = self._model_fit._params_trend
        self._variance = self._model_fit._params_variance
        self._coef_out = self._model_fit._params_ar
        self._coef_in = self._model_fit._params_exog
        self._coef_noise = self._model_fit._params_ma

    def predict(self, insample=True):
        """Prediction
        """
        if insample:
            # https://stackoverflow.com/questions/27931571/arma-predict-for-out-of-sample-forecast-does-not-work-with-floating-points
            foo = self._model_ext.get_prediction().predicted_mean.values
        else:
            foo = self._model_ext.get_prediction(dynamic=0).predicted_mean.values
            # raise NotImplementedError()
        return pd.DataFrame(foo, index=self.data_out.index, columns=self.data_out.columns)


#### Not Implemented ####
class VARMAX(Model_statsmodels):
    """VARMAX.

    Note
    ----
    Parameter estimation for VARMAX is very slow, and unstable according to the author of `statsmodels`.
    """
    def _fit(self, maxiter=10**3, **kwargs):
        self.model = varmax.VARMAX(self._Ytrn, self._Xtrn, order=self._order, **kwargs)
        self._model_fit = self.model.fit(maxiter=maxiter, **kwargs)
        raise NotImplementedError('VARMAX is not stable and not fully implemented.')

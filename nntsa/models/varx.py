"""
Collection of VARX (Vectorial Auto-Regressive with eXogeneous inputs) models.
"""

from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd

import sklearn
from sklearn import linear_model, metrics, pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn, optim, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split

from .. import tsa
from .. import stats
from . import mlp

import logging
logger = logging.getLogger(__name__)  # a named logger



class VARX(ABC):
    """Vectorial Auto-Regressive model with eXogeneous inputs (VARX).
    """

    @staticmethod
    def stack(data:pd.DataFrame, lag:int, offset:int=0) -> pd.DataFrame:
        """Build time-lagged data from original dataframe.

        This method augments the original dataframe by appending past values to the second axis. The appended columns have the name indicated by a suffix.

        Parameters
        ----------
        data
            input, a dataframe, the first axis is the time index.
        lag
            time lag, an integer.
        offset
            starting time offset, an integer, 0 by default.
        """
        foo = [data.shift(n+offset).values for n in range(lag)]
        cnames = []
        for n in range(lag):
            cnames += [f'{i}-{n+offset}' for i in data.columns]

        return pd.DataFrame(np.hstack(foo) if foo else [], index=data.index, columns=cnames)

    def __init__(self, data_in:pd.DataFrame, data_out:pd.DataFrame, order:tuple, *, train_idx:list=None, test_idx:list=None):
        """Initializer.

        Parameters
        ----------
        data_in
            input dataframe. The first axis is the time index.
        data_out
            output dataframe. It must share the same time index as `data_in`.
        order
            order of the VARX model, a tuple `(x, ar)` specifying the order of exogeneous and auto-regressive inputs.
        train_idx
            index of data used for training, a list, by default use all data.
        test_idx
            index of data used for testing, a list, by default use all data.
        """
        assert np.all(data_in.index == data_out.index)
        assert len(set(list(data_in.columns)+list(data_out.columns))) == len(data_in.columns)+len(data_out.columns), "Input and output dataframes must have distinct column names."

        # Forced conversion
        # without `.copy()` the dataframe created is just a reference
        self.data_in = pd.DataFrame(data_in).copy()
        self.data_out = pd.DataFrame(data_out).copy()
        self._order = order

        # Get explanatory variables, may contain nans
        self.data_in_stack = VARX.stack(self.data_in, self._order[0])
        self.data_out_stack = VARX.stack(self.data_out, self._order[1], 1)  # set offset=1 to exclude y[t] but keep y[t-1],...
        self.data_stack = self.data_in_stack.join(self.data_out_stack)  # full explanatory variables

        # Get nans-free explanatory and response variables
        # remove rows containing nans jointly
        # full data
        self._X, self._Y = tsa.drop_nan_rows(self.data_stack, self.data_out)

        # training data
        if train_idx is None:
            self._Xtrn, self._Ytrn = self._X, self._Y
        else:
            # trn_idx = [i for i in train_idx if i in self._Y.index]
            # self._Xtrn, self._Ytrn = self._X.loc[trn_idx], self._Y.loc[trn_idx]
            self._Xtrn, self._Ytrn = tsa.drop_nan_rows(self.data_stack.loc[train_idx], self.data_out.loc[train_idx])

        # testing data
        if test_idx is None:
            self._Xtst, self._Ytst = self._X, self._Y
        else:
            self._Xtst, self._Ytst = tsa.drop_nan_rows(self.data_stack.loc[test_idx], self.data_out.loc[test_idx])

        # variable for training prediction and r2 score
        self._prediction = None
        self._r2score = None

    @property
    def dim_in(self):
        """Number of exogeneous input variables.
        """
        return self.data_in.shape[1]

    @property
    def dim_out(self):
        """Number of output variables.
        """
        return self.data_out.shape[1]

    @property
    def dim_x(self):
        """Number of stacked variables used as input for ARX.
        """
        return self.data_in_stack.shape[1]

    @property
    def dim_ar(self):
        """Number of stacked variables used as input for ARX.
        """
        return self.data_out_stack.shape[1]

    @property
    def dim_stack(self):
        """Number of stacked variables used as input for ARX.
        """
        return self.data_stack.shape[1]

    def __str__(self):
        # name of the model
        # foo = 'ARX'
        # if np.squeeze(self.data_out.values).ndim > 1:
        #     # prefix for the vectorial case
        #     foo = 'V'+foo
        # return foo+f'{self._order}-'
        return f'VARX{self._order}-'

    @property
    def order_x(self) -> int:
        return self._order[0]

    @property
    def order_ar(self) -> int:
        return self._order[1]

    @property
    def is_static(self) -> bool:
        """Whether the model is time independent.
        """
        return self.order_x==1 and self.order_ar==0

    @property
    def is_linear(self) -> bool:
        """Wheather the model is linear.
        """
        if isinstance(self.basemodel, pipeline.Pipeline):
            return isinstance(self.basemodel.steps[-1][1], linear_model._base.LinearModel) & self._polynomial_degree == 1
        else:
            return isinstance(self.basemodel, linear_model._base.LinearModel)

    @property
    def coef(self) -> pd.DataFrame:
        return pd.concat([self.coef_x, self.coef_ar])

    @property
    def coef_x(self) -> pd.DataFrame:
        """Fitted coefficients of the inputs (exogeneous).
        """
        return pd.DataFrame(np.atleast_2d(self._coef_x), columns=self.data_out.columns, index=self.data_in_stack.columns)

    @property
    def coef_ar(self) -> pd.DataFrame:
        """Fitted coefficients of the outputs (auto-regressive).
        """
        return pd.DataFrame(np.atleast_2d(self._coef_ar), columns=self.data_out.columns, index=self.data_out_stack.columns)

    @property
    def intercept(self) -> pd.DataFrame:
        """Fitted intercept.
        """
        return pd.DataFrame(np.atleast_2d(self._intercept), columns=self.data_out.columns, index=['Intercept'])

    @abstractmethod
    def _fit(self, **kwargs):
        pass

    def fit(self, **kwargs):
        """Fit the model on training data then make prediction on the whole dataset.
        """
        # First fit the model.
        self._fit(**kwargs)
        # Make prediction on the whole dataset of initialization.
        self._prediction, _ = self.predict()  # the returned R2 score is dropped.
        _ = self.score(self._prediction)  # compute and save the R2 scores.

        # Construct the regression matrix in the linear case
        if self.is_linear:
            X = pd.DataFrame(np.zeros((1, self.data_stack.shape[1])), columns=self.data_stack.columns)
            b = self._predict(X)  # dim(b)==2
            A = []
            for n in range(X.shape[1]):
                X.iloc[:] = 0; X.iloc[0, n] = 1.
                v = self._predict(X)
                A.append(np.atleast_1d((v-b).squeeze()))
                # A.append(v-b)
            # self.basemodel.coef_ has shape (n_target, n_feature)
            self._coef_x, self._coef_ar = np.split(np.asarray(A), [self.dim_x], axis=0)
            self._intercept = np.asarray(b).squeeze()

        return self

    def debiasing(self, **kwargs):
        """Debiase a sparse model.

        Use this function after fitting e.g. a Lasso model.
        """
        coef0 = self.coef
        coef1 = coef0.copy() #; coef.loc['Intercep'] = 0
        intercept = pd.DataFrame(0, index=['Intercept'], columns=self.data_out.columns)

        for output in self.data_out.columns:
            nidx = (coef0[output].abs()>0).to_list()
            _Xtrn, _Ytrn = tsa.drop_nan_rows(self.data_stack.iloc[:,nidx], self.data_out[output])
            ols = linear_model.LinearRegression().fit(_Xtrn, _Ytrn)
            coef1.loc[nidx,output] = ols.coef_
            intercept[output] = ols.intercept_

        # assert ((coef0.abs()>0) ^ (coef1.abs()>0)).sum().sum() == 0
        return VARX_manual(coef1.loc[self.coef_x.index], coef1.loc[self.coef_ar.index], intercept, data_in=self.data_in, data_out=self.data_out, order=self._order)

    def score(self, prd:pd.DataFrame=None, **kwargs):
        """Compute the R2 scores of given predictions.

        Parameters
        ----------
        prd
            predictions. If not given it will be recomputed from data.

        Notes
        -----
        This method should be called only after `fit()`. It modifies the following inner variables:
        - `_r2score`: R2 of whole prediction
        - `_r2score_fit`: R2 of fit
        - `_r2score_val`: R2 of validation
        """
        # compute r2 score of the whole prediction
        if prd is None:
            prd, _ = self.predict()
        else:
            prd = self._prediction
        # self._prediction, _ = self.predict()

        self._r2score = stats.r2score(self.data_out.loc[self._Y.index], prd.loc[self._Y.index], **kwargs)
        # compute r2 score of fit using training index
        self._r2score_fit = stats.r2score(self.data_out.loc[self._Ytrn.index], prd.loc[self._Ytrn.index], **kwargs)
        # compute r2 score of validation using testing index
        self._r2score_val = stats.r2score(self.data_out.loc[self._Ytst.index], prd.loc[self._Ytst.index], **kwargs)

        return self._r2score

    @abstractmethod
    def _predict(self, x:pd.DataFrame, **kwargs) -> np.ndarray:
        pass

    def predict_insample(self, data_in:pd.DataFrame=None, data_out:pd.DataFrame=None) -> pd.DataFrame:
        """In-sample prediction.

        This is a supervised prediction and requires both the time-lagged inputs and outputs. The prediction length is bounded by the available inputs and outputs.

        Args
        ----
        data_in, data_out:
            input and output dataframe, unless both are provided the data of initialization will be used.
        """
        if data_in is None:
            # if the exogeneous input is not given, use data of initialization,
            data_in_stack = self.data_in_stack
        else:
            # otherwise prepare the input
            assert data_in.shape[1] == self.dim_in
            data_in_stack = VARX.stack(data_in, self.order_x)

        if data_out is None:
            data_out_stack = self.data_out_stack
        else:
            assert data_out.shape[1] == self.dim_out
            data_out_stack = VARX.stack(data_out, self.order_ar, 1)

        # prediction by the base model with all nans being filled
        data_stack0 = data_in_stack.join(data_out_stack)  # may contain nans
        data_stack = data_stack0.interpolate().bfill().ffill()
        goo = self._predict(data_stack)  # <- will often raise error if data contain nans

        # data_stack0 is returned for tracking invalid predictions.
        return pd.DataFrame(goo, index=data_in_stack.index, columns=self.data_out.columns), data_stack0

    def predict_outofsample(self, data_in:pd.DataFrame=None) -> pd.DataFrame:
        """Out-of-sample prediction.

        This is a non-supervised prediction and requires only the time-lagged inputs, but not the outputs. The prediction length is bounded only by the available inputs.

        Args
        ----
        data_in:
            input dataframe, if not provided the data of initialization will be used.
        """
        # def _predict_linear(data_in_stack):
        #     goo = np.dot(data_in_stack, self._coef_in) + self._intercept
        #     # # print(goo.shape, len(goo))
        #     for t in range(len(goo)):
        #         foo = np.asarray([goo[t-s-1] if t>s else np.zeros_like(goo[t]) for s in range(self.order_ar)]).ravel()
        #         # print(foo.shape)
        #         goo[t] += np.dot(self._coef_out.T, foo)
        #     return pd.DataFrame(goo, index=data_in_stack.index, columns=self.data_out.columns)

        if data_in is None:
            data_in_stack0 = self.data_in_stack
        else:
            assert data_in.shape[1] == self.dim_in
            data_in_stack0 = VARX.stack(data_in, self.order_x)

        data_in_stack = data_in_stack0.interpolate().bfill().ffill()
        # if self.is_linear:  # for linear model only
        #     return _predict_linear(data_in_stack)

        goo = np.zeros((len(data_in_stack), self.dim_out))
        for t in range(self.order_ar, len(goo)):
            xin = np.concatenate([data_in_stack.iloc[t].values, *goo[t-self.order_ar:t]]).reshape(1,-1)
            goo[t] = self._predict(pd.DataFrame(xin, columns=self.data_stack.columns))

        return pd.DataFrame(goo, index=data_in_stack.index, columns=self.data_out.columns), data_in_stack0

    def predict(self, data_in:pd.DataFrame=None, data_out:pd.DataFrame=None,
    *, fillna=False, **kwargs):
        """Make prediction.

        Parameters
        ----------
        data_in, data_out
            input and output dataframe, unless both are provided the data of initialization will be used.
        fillna
            whether to fill the missing values in the prediction, False by default. This alternates also the R2 score.
        kwargs
            keyword arguments for `stats.r2score()`.

        Returns
        -------
        prd, r2s
            prediction and R2 score.

        Note
        ----
        - R2 score is computed by truncating nans in both the prediction and the ground truth.
        - In-sample prediction is activated iff `data_in` and `data_out` are both given or both not given (use data of initialization in this case). Out-sample prediction is activated otherwise.
        - To activate out-of-sample prediction using data of initialization, set `data_in` to None and `data_out` to any value but None, e.g. use `predict(None, [])`. This way `data_out` will be ignored.
        - Missing values in `data_in` and `data_out` result in invalid local predictions. Therefore it is not necessary to fill nans beforehand.
        """
        # mode of prediction
        # (data_in is None) ^ (data_out is None) has 4 cases T^T=F^F=F, T^F=F^T=T
        # case T^T: insample prediction with default values
        # case F^F: insample prediction with ground truth
        # case T^F: outofsample prediction with default values, data_out being ignored
        # case F^T: outofsample prediction without ground truth
        outofsample = (data_in is None) ^ (data_out is None)

        if outofsample:
            prd, exg = self.predict_outofsample(data_in)
        else:
            prd, exg = self.predict_insample(data_in, data_out)

        if not fillna:
            # `exg` may contain nans (due to e.g. data stacking of `data_in` and `data_out`). They are filled before making prediction so that `prd` does not contain nans. For accurate R2 score, these nans need to be restored and their positions will be removed in the computation.
            # This is similar to the treatement applied on `self._Xtrn, self._Ytrn`.
            na_idx = exg.isna().prod(axis=1)==1
            prd.loc[na_idx] = np.nan

        # compute R2 score of prediction with truncation of missing values
        if outofsample:
            # valid only if data_in is not given, since otherwise there is no ground truth to compare to.
            r2s = stats.r2score(*tsa.drop_nan_rows(self.data_out, prd), **kwargs) if data_in is None else None
        else:
            r2s = stats.r2score(*tsa.drop_nan_rows(self.data_out if data_out is None else data_out, prd), **kwargs)

        return prd, r2s


class VARX_sklearn(VARX):
    """VARX based on solvers of the package `scikit-learn`.
    """
    def __init__(self, *args, basemodel=linear_model.LinearRegression(), **kwargs):
        """
        Parameters
        ----------
        model
            model of regression, an object of `sklearn` regressor.
        """
        super().__init__(*args, **kwargs)

        # Sanity check: model must be a sklearn regressor.
        # pipeline
        if isinstance(basemodel, pipeline.Pipeline):
            if len(basemodel) > 3:
                raise Exception("Unsupported pipeline: only the compound model 'StandarScaler-PolynomialFeature-Estimator' can be used.")

            if len(basemodel) == 2:
                assert isinstance(basemodel.steps[0][1], sklearn.preprocessing._data.StandardScaler) | isinstance(basemodel.steps[0][1], sklearn.preprocessing._polynomial.PolynomialFeatures), "First step in the pipeline must be a StandardScaler or PolynomialFeature."

            # polynomial degree of the pipeline model
            for _, model in basemodel.steps:
                if isinstance(model, sklearn.preprocessing._polynomial.PolynomialFeatures):
                    self._polynomial_degree = model.degree
                    break
            else:
                self._polynomial_degree = 1

            # assert isinstance(basemodel, sklearn.base.BaseEstimator)  # True also for PolynomialFeatures
            assert hasattr(basemodel.steps[-1][1], 'predict'), "Last step in the pipeline must be an estimator."

            self._base_estimator = basemodel.steps[-1][1]
        else:
            assert hasattr(basemodel, 'predict')
            self._base_estimator = basemodel

        self.basemodel = basemodel

    def __str__(self):
        return super().__str__()+self.basemodel_str

    @property
    def basemodel_str(self):
        try:
            return self._basemodel_str
        except:
            return str(self.basemodel)
        # return str(self.basemodel).split('(')[0]

    @basemodel_str.setter
    def basemodel_str(self, s:str):
        self._basemodel_str = s

    def _fit(self, **kwargs):
        """Fit the model.
        """
        # Warning will be raised if the feature names at fitting and at prediction do not agree.
        # fit with feature names
        self.basemodel.fit(self._Xtrn, self._Ytrn, **kwargs)
        # # fit without feature names
        # self.basemodel.fit(self._Xtrn.values, self._Ytrn.values, **kwargs)

    def _predict(self, x:pd.DataFrame) -> np.ndarray:
        # `model.predict` does the following:
        # goo = data @ model.coef_.T + model.intercept_
        # See:
        # https://github.com/scikit-learn/scikit-learn/blob/9b033758e/sklearn/linear_model/_base.py#L348
        return self.basemodel.predict(x)


class VARX_manual(VARX):
    """VARX model constructed from coefficients.
    """
    def __init__(self, coef_x, coef_ar, intercept, *args, **kwargs):
        """
        Parameters
        ----------
        coef_x
            coefficients of exogeneous input, shape (output, input)
        coef_ar
            coefficients of autoregression
        intercept
            coefficients of intercept
        """
        super().__init__(*args, **kwargs)
        self._coef_x = np.asarray(coef_x) #.squeeze()
        self._coef_ar = np.asarray(coef_ar) #.squeeze()
        self._intercept = np.asarray(intercept) #.squeeze()
        self._coef = np.vstack([self._coef_x, self._coef_ar])

    def _fit(self):
        raise NotImplementedError()

    def _predict(self, x: pd.DataFrame, **kwargs) -> np.ndarray:
        goo = np.dot(x, self._coef) + self._intercept
        # return pd.DataFrame(goo, index=x.index, columns=self.data_out.columns)
        return goo

"""
MLP regression.
"""

import numpy as np
import pandas as pd

import torch
from torch import nn, optim, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader #, random_split
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from sklearn import model_selection, metrics

# https://scikit-learn.org/stable/developers/develop.html
# https://scikit-learn.org/stable/modules/classes.html


import logging
logger = logging.getLogger(__name__)  # a named logger


class MLP_Regression(nn.Module):
    """Multilayer perceptrons for regression.
    """
    def __init__(self, n_input:int, n_output:int, *, n_hidden:list[int]=[], bias:bool=True, activ:list=[]) -> None:
        """MLP_Regression initialization.

        Parameters
        ----------
        n_input
            number of input variables.
        n_output
            number of output variables.
        n_hidden
            number of units in the hidden layers, no hidden layers by default.
        bias, optional
            add bias to the hidden layers, True by default.
        activ, optional
            activation function for hidden layers. Use `nn.ReLU` by default.
        """
        super().__init__()

        n_units = [n_input, *n_hidden, n_output]
        Layers = []

        for l, n_in in enumerate(n_units[:-1]):
            layer = nn.Linear(n_in, n_units[l+1], bias=bias)
            # use xavier initialization for regression
            nn.init.xavier_uniform_(layer.weight)
            Layers.append(layer)
            if l < len(n_units)-2:
                # no activation for the last layer
                try:
                    Layers.append(activ[l])
                except:
                    Layers.append(nn.ReLU())

        self.layers = nn.Sequential(*Layers)

    def forward(self, x):
        return self.layers(x)

    @property
    def number_parameters(self) -> int:
        """Total number of parameters of the MLP model.
        """
        n_parms = 0
        for p in self.parameters():
            n_parms += np.prod(p.size())
        return n_parms


class MLPRegressor(BaseEstimator, RegressorMixin):
    """MLP based regressor.

    Notes
    -----
    Methods in `sklearn.pipeline` require that all variables must be saved with their given names.
    """
    def __init__(self, n_hidden:list[int], bias=True, *, loss=nn.MSELoss, loss_parms:dict={}, reg_parms={'weight':10**-3, 'pow':2.}, activ=nn.ReLU, activ_parms:dict={}, optim:callable=optim.Adam, optim_parms:dict={}) -> None:
        """_summary_

        Parameters
        ----------
        n_hidden
            _description_
        bias, optional
            _description_, by default True
        loss, optional
            _description_, by default nn.MSELoss
        loss_parms, optional
            _description_, by default {}
        reg_parms, optional
            _description_, by default {'weight':10**-3, 'pow':2.}
        activ, optional
            _description_, by default nn.ReLU
        activ_parms, optional
            _description_, by default {}
        optim, optional
            _description_, by default optim.Adam
        optim_parms, optional
            _description_, by default {}
        """
        # self.model = MLP_Regression(*args, **kwargs)
        self.n_hidden = n_hidden
        self.bias = bias
        self.activ = activ
        self.activ_parms = activ_parms
        self.optim = optim
        self.optim_parms = optim_parms
        self.loss = loss(**loss_parms)
        self.loss_parms = loss_parms
        self.reg_parms = reg_parms
        self.model = None

    def __str__(self) -> str:
        # Gotcha: f-string does not allow backslash!
        # return f"MLPRegressor(hidden={self.n_hidden}, bias={self.bias}, activ={str(self.activ).split('.')[-1].split('\'')[0]}, loss={self.loss}, reg={self.reg_parms})"
        return "MLPRegressor(hidden={}, bias={}, activ={}, loss={}, reg={})".format(self.n_hidden, self.bias, str(self.activ).split('.')[-1].split('\'')[0], str(self.loss).split('(')[0], self.reg_parms)

    @property
    def name(self) -> str:
        # return "MLPRegressor({}, {}, {}, {})".format(self.n_hidden, str(self.activ).split('.')[-1].split('\'')[0], str(self.loss).split('(')[0], self.reg_parms)
        return "MLPRegressor({}, {})".format(self.n_hidden, str(self.activ).split('.')[-1].split('\'')[0])

    def init_model(self, n_input, n_output):
        self.model = MLP_Regression(n_input, n_output, n_hidden=self.n_hidden,
            activ=[self.activ(**self.activ_parms) for _ in self.n_hidden])
        self.solver = self.optim(self.model.parameters(), **self.optim_parms)  # optimizer

    def fit(self, X, Y, *,
        # Xtrn:np.ndarray, Ytrn:np.ndarray, *, Xtst:np.ndarray=None, Ytst:np.ndarray=None,
            n_epochs:int=10**3, batch_size:int=32, shuffle:bool=True, split_parms=dict(test_size=0.25, random_state=1234), test_step:int=10):
        """Fit the MLP regressor model.

        Parameters
        ----------
        X, Y
            input and output data.
        n_epochs
            number of training epochs.
        batch_size, shuffle
            for training data only.
        split_parms
            keyword parameters for `sklearn.model_selection.train_test_split()`.
        test_step
            period of evaluation and logging.
        """
        if self.model is None:
            # self.init_model(Xtrn.shape[1], Ytrn.shape[1])
            self.init_model(X.shape[1], Y.shape[1])

        Xtrn, Xtst, Ytrn, Ytst = model_selection.train_test_split(np.asarray(X), np.asarray(Y), **split_parms)
        train_dl = DataLoader(TensorDataset(Tensor(np.asarray(Xtrn)), Tensor(np.asarray(Ytrn))), batch_size=batch_size, shuffle=shuffle)
        test_dl = DataLoader(TensorDataset(Tensor(np.asarray(Xtst)), Tensor(np.asarray(Ytst))), batch_size=batch_size, shuffle=False)
        reg_weight = self.reg_parms['weight']
        reg_pow = self.reg_parms['pow']

        n_parameters = self.model.number_parameters  # total number of parameters in NN

        res = []

        for epoch in range(n_epochs):
            # set the training mode
            self.model.train()
            # enumerate mini batches
            for i, (x, y) in enumerate(train_dl):
                # clear the gradients
                self.solver.zero_grad()
                # compute the model output and loss
                loss = self.loss(self.model(x), y)

                # compute loss + reg
                if reg_weight > 0:
                    val = 0.
                    for p in self.model.parameters():
                        val += p.abs().pow(reg_pow).sum()
                    if self.loss.reduction == 'mean':
                        val /= n_parameters
                        full_loss = loss + reg_weight * val #.pow(1/reg_pow)
                else:
                    full_loss = loss

                # credit assignment
                full_loss.backward()
                # update model weights
                self.solver.step()

            # set the evaluation mode
            if (test_step > 0) & (Xtst is not None) & (Ytst is not None):
                if epoch % test_step==0:
                    # https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval
                    self.model.eval()
                    with torch.no_grad():
                        # mse = nn.functional.mse_loss(self.model(Tensor(np.asarray(Xtst))).detach(), Tensor(np.asarray(Ytst)))
                        mse = self.evaluate(test_dl)
                        res.append((full_loss.numpy(), loss.numpy(), mse.numpy()))
                        logger.info(f"Epoch {epoch}: full_loss={full_loss:.3e}, loss={loss:.3e}, eval.mse={mse:.3e}")
            else:
                res.append((full_loss.numpy(), loss.numpy()))
                logger.info(f"Epoch {epoch}: full_loss={full_loss:.3e}, loss={loss:.3e}")
                # logger.info(f"Epoch {epoch}: loss={loss:.3e}")

        self._fit_info = np.asarray(res)

    def evaluate(self, test_dl) -> float:
        """Evaluate the model.

        Parameters
        ----------
        test_dl
            Dataloader of the test data.

        Returns
        -------
            MSE of test.
        """
        prd, obs = [], []
        for i, (x, y) in enumerate(test_dl):
            prd.append(self.model(x).detach().numpy())
            obs.append(y.numpy())
            # obs.append(y.numpy().reshape((-1, 1)))

        # mse = metrics.mean_squared_error(obs, prd)
        mse = nn.functional.mse_loss(Tensor(np.vstack(prd)), Tensor(np.vstack(obs)))
        return mse

    def predict(self, X:np.ndarray):
        """Predict with the model.
        """
        with torch.no_grad():
            Y = self.model(Tensor(np.asarray(X))).detach().numpy()

        return Y

    def get_params(self, **kwargs):
        return [p.detach().numpy() for p in self.model.parameters()]

    def set_params(self, *args, **kwargs):
        with torch.no_grad():
            for n, layer in enumerate(self.layers):
                try:
                    layer.weight = nn.Parameter(torch.Tensor(args[2*n]))
                    layer.bias = nn.Parameter(torch.Tensor(args[2*n+1]))
                except:
                    break


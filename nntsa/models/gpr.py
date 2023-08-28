import numpy as np
import pandas as pd

import gpflow
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from sklearn import model_selection, metrics

import logging
logger = logging.getLogger(__name__)  # a named logger


class GPRegressor(BaseEstimator, RegressorMixin):
  """Gaussian Process regressor.
  """
  def __init__(self, *args, kfunc=gpflow.kernels.Matern32(), mfunc=None, **kwargs):
    """
    Args
    ----
    kfunc:
        kernel function
    mfunc:
        mean function
    """
    # super().__init__(*args, **kwargs)
    assert self.dim_out==1, "Only scalar output is supported."

    self.kfunc = kfunc
    self.mfunc = mfunc

  def _fit(self, *, opt=gpflow.optimizers.Scipy(), maxiter=10**3, **kwargs):
    """Fit the model.
    """
    gpr = gpflow.models.GPR(data=(self._Xtrn.values, np.atleast_1d(self._Ytrn.values.squeeze())), kernel=self.kfunc, mean_function=self.mfunc)
    opt_logs = opt.minimize(gpr.training_loss, gpr.trainable_variables, options=dict(maxiter=maxiter))
    # print_summary(gpr)
    self.basemodel = gpr

  def _predict(self, x:pd.DataFrame) -> np.ndarray:
    self._predict_mean, self._predict_var = self.basemodel.predict_y(x.values)
    # dm = pd.Series(moo.numpy().squeeze(), index=x.index, name=self.data_out.columns[0]).to_frame()
    # self._predict_dv = pd.Series(voo.numpy().squeeze(), index=x.index, name=self.data_out.columns[0]).to_frame()
    return self._predict_mean

  def __init__(self, **kwargs) -> None:
    """
    Notes
    -----
    Methods in `sklearn.pipeline` require that all variables must be saved with their given names.
    """
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
      activ=[self.activ(**self.activ_parms) for _ in range(len(self.n_hidden))])
      self.solver = self.optim(self.model.parameters(), **self.optim_parms)

  def fit(self, X, Y, *,
  # Xtrn:np.ndarray, Ytrn:np.ndarray, *, Xtst:np.ndarray=None, Ytst:np.ndarray=None,
  n_epochs=10**3, batch_size=32, shuffle=True, split_parms=dict(test_size=0.25, random_state=1234), test_step=10):
    """
    Args
    ----
    X, Y: input and output data.
    n_epochs: number of training epochs.
    batch_size, shuffle: for training data only.
    split_parms: keyword parameters for `sklearn.model_selection.train_test_split()`, a dictionary.
    test_step: evaluate and log every `test_step`.
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

  def evaluate(self, test_dl):
    prd, obs = [], []
    for i, (x, y) in enumerate(test_dl):
      prd.append(self.model(x).detach().numpy())
      obs.append(y.numpy())
      # obs.append(y.numpy().reshape((-1, 1)))

    # mse = metrics.mean_squared_error(obs, prd)
    mse = nn.functional.mse_loss(Tensor(np.vstack(prd)), Tensor(np.vstack(obs)))
    return mse

  def predict(self, X:np.ndarray):
    with torch.no_grad():
      Y = self.model(Tensor(np.asarray(X))).detach().numpy()

    return Y

  # def get_params(self, **kwargs):
  #   return [p.detach().numpy() for p in self.model.parameters()]

  # def set_params(self, *args, **kwargs):
  #   with torch.no_grad():
  #     for n, layer in enumerate(self.layers):
  #       try:
  #         layer.weight = nn.Parameter(torch.Tensor(args[2*n]))
  #         layer.bias = nn.Parameter(torch.Tensor(args[2*n+1]))
  #       except:
  #         break


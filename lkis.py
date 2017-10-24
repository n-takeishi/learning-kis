#!/usr/bin/env python
# coding: utf-8

""" Learning Koopman Invariant Subspace
 (c) Naoya Takeishi, 2017.
 takeishi@ailab.t.u-tokyo.ac.jp
"""

import numpy
from scipy import linalg
from chainer import link
from chainer import Variable
from chainer import Chain
from chainer import dataset
from chainer import reporter as reporter_module
from chainer import training
from chainer import initializers
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

# ==========

def ls_solution(g0, g1):
  """ Get least-squares solution matrix for regression from rows of g0
      to rows of g1. Both g0 and g1 are chainer's Variable.
  """
  g0t = F.transpose(g0)
  if g0.shape[0] >= g0.shape[1]:
    g0pinv = F.matmul(F.inv(F.matmul(g0t, g0)), g0t)
  else:
    g0pinv = F.matmul(g0t, F.inv(F.matmul(g0, g0t)))
  K = F.transpose(F.matmul(g0pinv, g1))
  return K

# ==========

def dmd(y0, y1, eps=1e-6):
  """ Do DMD. Both y0 and y1 are numpy array.
  """
  Y0 = y0.T
  Y1 = y1.T
  U, S, Vh, = linalg.svd(Y0, full_matrices=False)
  r = len(numpy.where(S>=eps)[0])
  U = U[:,:r]
  invS = numpy.diag(1./S[:r])
  V = Vh.conj().T[:,:r]
  M = numpy.dot(numpy.dot(Y1, V), invS)
  A_til = numpy.dot(U.conj().T, M)
  lam, z_til, w_til = linalg.eig(A_til, left=True)
  w = numpy.dot(numpy.dot(M, w_til), numpy.diag(1./lam)) + 1j*numpy.zeros(z_til.shape)
  z = numpy.dot(U, z_til) + 1j*numpy.zeros(z_til.shape)
  for i in range(w.shape[1]):
    z[:,i] = z[:,i] / numpy.dot(w[:,i].conj(), z[:,i])
  return lam, w, z

# ==========

class DelayPairDataset(dataset.DatasetMixin):
  def __init__(self, values, dim_delay, n_lag=1):
    if isinstance(values, list):
      self.values = values
    else:
      self.values = [values,]
    self.lens = tuple(value.shape[0]-(dim_delay-1)*n_lag-1 for value in self.values)
    self.a_s = [0 for i in range(sum(self.lens))]
    for i in range(sum(self.lens)):
      for j in range(len(self.values)):
        if i >= sum(self.lens[0:j]):
          self.a_s[i] = j
    self.dim_delay = dim_delay
    self.n_lag = n_lag

  def __len__(self):
    return sum(self.lens)

  def get_example(self, i):
    tau = self.n_lag
    k = self.dim_delay
    a = self.a_s[i]
    b = i - sum(self.lens[0:a])
    return (self.values[a][b:b+(k-1)*tau+1:tau], self.values[a][b+1:b+(k-1)*tau+2:tau])

# ==========

class Embedder(Chain):
  def __init__(self, dimy, delay, dim_emb):
    super(Embedder, self).__init__(l1 = L.Linear(dimy*delay, dim_emb))

  def __call__(self, x):
    return self.l1(x)

# ==========

class Observable(Chain):
  def __init__(self, dim_g, dim_emb):
    n_h = round((dim_g+dim_emb)*0.5)
    super(Observable, self).__init__(
      l1 = L.Linear(dim_emb, n_h),
      p1 = L.PReLU(),
      b1 = L.BatchNormalization(n_h),
      l2 = L.Linear(n_h, dim_g)
    )
    self.add_persistent('dim_g', dim_g)

  def __call__(self, x, train=True):
    return self.l2(self.b1(self.p1(self.l1(x)), test=not train))

# ==========

class Reconstructor(Chain):
  def __init__(self, dim_y, dim_g):
    n_h = round((dim_y+dim_g)*0.5)
    super(Reconstructor, self).__init__(
      l1 = L.Linear(dim_g, n_h),
      p1 = L.PReLU(),
      b1 = L.BatchNormalization(n_h),
      l2 = L.Linear(n_h, dim_y)
    )

  def __call__(self, x, train=True):
    # The nonlinearlity of Reconstructor is realized by p1 (PReLU function),
    # so eliminating p1 from calculation makes Reconstructor linear.
    #return self.l2(self.b1(self.l1(x), test=not train))
    return self.l2(self.b1(self.p1(self.l1(x)), test=not train))

# ==========

class Network(Chain):
  def __init__(self, dim_emb, dim_g, dim_y):
    super(Network, self).__init__(
      b = L.BatchNormalization(dim_emb),
      g = Observable(dim_g, dim_emb),
      h = Reconstructor(dim_y, dim_g)
    )

  def __call__(self, y0, y1, phi=None, train=True):
    x0 = self.b(phi(y0), test=not train)
    x1 = self.b(phi(y1), test=not train)
    g0 = self.g(x0, train=train)
    g1 = self.g(x1, train=train)
    h0 = self.h(g0, train=train)
    h1 = self.h(g1, train=train)
    return g0, g1, h0, h1

# ==========

class Loss(Chain):
  def __init__(self, phi, net, alpha=1.0, decay=0.9):
    super(Loss, self).__init__(
      phi = phi,
      net = net
    )
    self.add_persistent('alpha', alpha)
    self.add_persistent('decay', decay)

  def __call__(self, y0, y1, train=True):
    g0, g1, h0, h1 = self.net(y0, y1, phi=self.phi, train=train)

    loss1 = F.mean_squared_error(F.linear(g0, ls_solution(g0, g1)), g1)
    loss2 = F.mean_squared_error(h0, F.transpose(y0,axes=(1,0,2))[-1])
    loss3 = F.mean_squared_error(h1, F.transpose(y1,axes=(1,0,2))[-1])
    loss = loss1 + self.alpha*0.5*(loss2+loss3)

    reporter_module.report({
      'loss': loss,
      'loss_kpm': loss1,
      'loss_rec': 0.5*(loss2+loss3)
    }, self.net)

    return loss

# ==========

class Updater(training.StandardUpdater):
  def update_core(self):
    batch = self._iterators['main'].next()
    in_arrays = self.converter(batch, self.device)
    in_vars = tuple(Variable(x) for x in in_arrays)
    for optimizer in self._optimizers.values():
      optimizer.update(self.loss_func, *in_vars)

# ==========

class Evaluator(extensions.Evaluator):
  def __init__(self, iterator, target, converter=dataset.convert.concat_examples,
               device=None, eval_hook=None, eval_func=None, trigger=(1,'epoch')):
    if isinstance(iterator, dataset.iterator.Iterator):
      iterator = {'main': iterator}
    self._iterators = iterator

    if isinstance(target, link.Link):
      target = {'main': target}
    self._targets = target

    self.converter = converter
    self.device = device
    self.eval_hook = eval_hook
    self.eval_func = eval_func
    self.trigger = trigger

  def evaluate(self):
    iterator = self._iterators['main']
    target = self._targets['main']
    eval_func = self.eval_func or target

    if self.eval_hook:
      self.eval_hook(self)

    if hasattr(iterator, 'reset'):
      iterator.reset()
      it = iterator
    else:
      it = copy.copy(iterator)

    summary = reporter_module.DictSummary()
    for batch in it:
      observation = {}
      with reporter_module.report_scope(observation):
        in_arrays = self.converter(batch, self.device)
        in_vars = tuple(Variable(x, volatile='on')
                        for x in in_arrays)
        eval_func(*in_vars, train=False)
        summary.add(observation)

    return summary.compute_mean()

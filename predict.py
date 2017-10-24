#!/usr/bin/env python
# coding: utf-8

""" Learning Koopman Invariant Subspace
 (c) Naoya Takeishi, 2017.
 takeishi@ailab.t.u-tokyo.ac.jp
"""

import numpy as np
np.random.seed(1234567890)

import os, argparse, json
from scipy.io import savemat

from chainer import Variable
from chainer import iterators
from chainer import dataset as dataset_module
from chainer import serializers
from chainer import cuda

import lkis

# -- Parse arguments
parser = argparse.ArgumentParser(description='Learning Koopman Invariant Subspace')
parser.add_argument('name', nargs=None, type=str, help='name of experiment')
parser.add_argument('--rootdir', default='./exp_$name$/', type=str,
          help='root directory of experiment (./exp_$name$/)')
parser.add_argument('--outputdir', '-o', default='result/', type=str,
          help='output directory (result/)')
parser.add_argument('--prefix', '-p', default='test', type=str,
          help='prefix of data file (test)')
parser.add_argument('--numtest', '-t', default=1, type=int,
          help='number of test datasets (1)')
parser.add_argument('--horizon', '-z', default=10, type=int,
          help='prediction horizon timestep (10)')
parser.add_argument('--save', '-s', default=False, action='store_true',
          help='save prediction results (False)')

args = parser.parse_args()

args.rootdir = args.rootdir.replace('$name$', args.name)
args.outputdir = os.path.join(args.rootdir, args.outputdir)

# -- Load arguments at testing
with open(os.path.join(args.outputdir,'args.json'), 'r') as jsonfile:
  args_json = json.load(jsonfile)

args.datadir = args_json['datadir']
args.preprocessing = args_json['preprocessing']
args.delay = args_json['delay']
args.dimy = args_json['dimy']
args.dimemb = args_json['dimemb']
args.dimobs = args_json['dimobs']
args.alpha = args_json['alpha']

# -- Make result directory if it does not exist
if not os.path.isdir(args.outputdir):
  os.makedirs(args.outputdir)

# -- Check fixed or not
assert not os.path.isfile(os.path.join(args.outputdir,'fix_output')), 'output is fixed'

# -- Load test data
if args.numtest>1:
  data_test = [np.loadtxt(os.path.join(args.datadir,'test_%d.txt' % i),
                           dtype=np.float32, delimiter=',', ndmin=2)
                for i in range(args.numtest)]
else:
  try:
    data_test = [np.loadtxt(os.path.join(args.datadir,'test_0.txt'),
                             dtype=np.float32, delimiter=',', ndmin=2),]
  except FileNotFoundError:
    data_test = [np.loadtxt(os.path.join(args.datadir,'test.txt'),
                             dtype=np.float32, delimiter=',', ndmin=2),]
  args.numtest = 1

data_test_all = np.concatenate(data_test, axis=0)

# -- Preprocessing
if args.preprocessing:
  train_absmax = np.load(os.path.join(args.outputdir, 'train_absmax'))
else:
  train_absmax = np.ones((1,data_test_all.shape[1]), dtype=np.float32)
for i in range(args.numtest):
  data_test[i] = data_test[i] / train_absmax

data_test_all = np.concatenate(data_test, axis=0)

# -- Show settings
print('Prediction by LKIS-DMD')
print()
print('experiment name       : %s' % args.name)
for i in range(args.numtest):
  print('  --> test data size   = (%d, %d)' %
        (data_test[i].shape[0], data_test[i].shape[1]))

# -- Load trained model
loss = lkis.Loss(lkis.Embedder(args.dimy, args.delay, args.dimemb),
                 lkis.Network(args.dimemb, args.dimobs, args.dimy), args.alpha)
serializers.load_npz(os.path.join(args.outputdir, 'train_model_final.npz'), loss)
print('(%s) loaded the trained model in %s' % (args.name, args.outputdir))

# -- Load trained DMD
lam = np.load(os.path.join(args.outputdir, 'train_lam.npy'))
w = np.load(os.path.join(args.outputdir, 'train_w.npy'))
z = np.load(os.path.join(args.outputdir, 'train_z.npy'))

# -- Make datasets
dataset_test = lkis.DelayPairDataset(data_test, args.delay)

# -- Set iterators
test_iter = iterators.SerialIterator(dataset_test,
                                     batch_size=len(dataset_test),
                                     repeat=False, shuffle=False)

# -- Apply the learned network to the test data and save the results
def save_results(dataset, saveflag, filename, lam, w, z):
  for n in range(len(dataset.lens)):
    in_vars = tuple(Variable(x, volatile='on')
                    for x in dataset_module.concat_examples(
                      dataset[sum(dataset.lens[0:n]):sum(dataset.lens[0:n+1])]))
    g0, g1, h0, h1 = loss.net(*in_vars, phi=loss.phi, train=False)

    # prepare outputs
    y0 = np.transpose(in_vars[0].data, axes=(1,0,2))[-1]
    y1 = np.transpose(in_vars[1].data, axes=(1,0,2))[-1]
    g0 = cuda.to_cpu(g0.data); g1 = cuda.to_cpu(g1.data)
    h0 = cuda.to_cpu(h0.data); h1 = cuda.to_cpu(h1.data)

    # do prediction in g-space
    g_preds = [] # starts from prediction corresponding to g1
    for j in range(args.horizon):
      tmp = np.zeros(g0.shape) + 1j*np.zeros(g0.shape)
      if j==0:
        g_prev = g0
      else:
        g_prev = g_preds[j-1]
      for i in range(g1.shape[0]):
        tmp[i] = np.sum(np.dot(w, np.diag(np.dot(np.diag(lam), np.dot(z.conj().T, g_prev[i])))), axis=1)
      g_preds.append(tmp.copy())

    # back-project the prediction to y-space
    y_preds = []
    rmses = [0 for i in range(args.horizon)]
    for j in range(args.horizon):
      g_pred_var = Variable(np.real(g_preds[j]).astype(dtype=np.float32), volatile='on')
      tmp = loss.net.h(g_pred_var, train=False).data
      y_preds.append(tmp.copy())
      if j==0:
        tmp_end = tmp.shape[0]
      else:
        tmp_end = -j
      rmses[j] = np.sqrt(np.mean(np.square(tmp[:tmp_end]-y1[j:])))

    # save
    if saveflag:
      savemat(os.path.join(args.outputdir, '{0}_{1:d}'.format(filename, n)),
              {'lam':lam, 'w':w, 'z':z, 'prediction_rmses':rmses,
               'phi_W':loss.phi.l1.W.data, 'phi_b':loss.phi.l1.b.data,
               'g_preds':g_preds, 'y_preds':y_preds,
               'y0':y0, 'g0':g0, 'h0':h0, 'y1':y1, 'g1':g1, 'h1':h1
              })

  return rmses

rmses = save_results(dataset_test, args.save, 'output_%s' % args.prefix, lam, w, z)
if args.save:
  print('(%s) saved the computed quantities on %s data in %s' % (args.name, args.prefix, args.outputdir))

# -- Prediction
d = args.dimy
n = args.delay*d
x = args.dimemb
g = args.dimobs
h1 = round((x+g)/2.0)
h2 = round((g+args.dimy)/2.0)
c = (n*x+x) + (x*h1+h1+h1*g+g) + (g*h2+h2+h2*d+d)
print()
print('# of parameters of the model = %d + %d' % (c,g*g))
print('Prediction: RMSE (test)       : horizon = 1 ... %d' % args.horizon)
print(rmses)

#!/usr/bin/env python
# coding: utf-8

""" Learning Koopman Invariant Subspace
 (c) Naoya Takeishi, 2017.
 takeishi@ailab.t.u-tokyo.ac.jp
"""

import numpy as np
np.random.seed(1234567890)
import chainer
if chainer.cuda.available:
  chainer.cuda.cupy.random.seed(1234567890)

import os, argparse, json, time
from scipy.io import savemat

from chainer import Variable
from chainer import iterators
from chainer import optimizers
from chainer import optimizer as optimizer_module
from chainer import dataset as dataset_module
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import cuda

import lkis

# -- Parse arguments
parser = argparse.ArgumentParser(description='Learning Koopman Invariant Subspace')
parser.add_argument('name', nargs=None, type=str, help='name of experiment')
parser.add_argument('--resume', '-r', default=None, type=str,
          help='Resume the optimization from snapshot')
parser.add_argument('--rootdir', default='./exp_$name$/', type=str,
          help='root directory of experiment (./exp_$name$/)')
parser.add_argument('--datadir', default='data/', type=str,
          help='data directory (data/)')
parser.add_argument('--outputdir', '-o', default='result/', type=str,
          help='output directory (result/)')
parser.add_argument('--numtrain', '-n', default=1, type=int,
          help='number of training datasets (1)')
parser.add_argument('--numval', '-v', default=-1, type=int,
          help='number of validation datasets (non-positive value indicates no validation) (-1)')
parser.add_argument('--preprocessing', default=False, action='store_true',
          help='do preprocessing if specified')
parser.add_argument('--gpu', default=-1, type=int,
          help='GPU ID (negative value indicates CPU) (-1)')
parser.add_argument('--batchsize', '-m', default=-1, type=int,
          help='minibatch size (non-positive value indicates full-batch) (-1)')
parser.add_argument('--epoch', '-e', default=1000, type=int,
          help='number of epochs to learn (1000)')
parser.add_argument('--learning_rate', '-l', default=1e-3, type=float,
          help='learning rate (1e-3)')
parser.add_argument('--fixed_rate', default=False, action='store_true',
          help='fix the learning rate (False)')
parser.add_argument('--add_epoch', default=1000, type=int,
          help='number of additional epochs to learn only reconstructor (1000)')
parser.add_argument('--add_learning_rate', default=None, type=float,
          help='learning rate in the additional epochs (same with learning_rate)')
parser.add_argument('--delay', '-k', default=2, type=int,
          help='number of delays (2)')
parser.add_argument('--dimemb', '-x', default=None, type=int,
          help='dimension of embedding (dimy*delay)')
parser.add_argument('--dimobs', '-g', default=None, type=int,
          help='dimension of observable (dimemb)')
parser.add_argument('--alpha', '-a', default=1e-2, type=float,
          help='coefficient of reconstruction regularizer (1e-2)')
parser.add_argument('--beta', '-b', default=0.0, type=float,
          help='coefficient of lasso regularizer (0)')
parser.add_argument('--gamma', '-c', default=0.0, type=float,
          help='coefficient of ridge regularizer (0)')
parser.add_argument('--snapshot_interval', '-s', default=-1, type=int,
          help='number of interval epochs to save snapshots (-1)')
parser.add_argument('--fixed_embedder', default=False, action='store_true',
          help='fix the parameter of the embedder (False)')
parser.add_argument('--log_in_iteration', default=False, action='store_true',
          help='Report log file in each iteration (False)')

args = parser.parse_args()

args.rootdir = args.rootdir.replace('$name$', args.name)
args.datadir = os.path.join(args.rootdir, args.datadir)
args.outputdir = os.path.join(args.rootdir, args.outputdir)

# -- Make result directory if it does not exist
if not os.path.isdir(args.outputdir):
  os.makedirs(args.outputdir)

# -- Check fixed or not
assert not os.path.isfile(os.path.join(args.outputdir,'fix_output')), 'output is fixed'

# -- Load training data
if args.numtrain>1:
  data_train = [np.loadtxt(os.path.join(args.datadir,'train_%d.txt' % i),
                           dtype=np.float32, delimiter=',', ndmin=2)
                for i in range(args.numtrain)]
else:
  try:
    data_train = [np.loadtxt(os.path.join(args.datadir,'train_0.txt'),
                             dtype=np.float32, delimiter=',', ndmin=2),]
  except FileNotFoundError:
    data_train = [np.loadtxt(os.path.join(args.datadir,'train.txt'),
                             dtype=np.float32, delimiter=',', ndmin=2),]
  args.numtrain = 1

data_train_all = np.concatenate(data_train, axis=0)

# -- Load validation data if any
if args.numval>1:
  data_val = [np.loadtxt(os.path.join(args.datadir,'val_%d.txt' % i),
                         dtype=np.float32, delimiter=',', ndmin=2)
                for i in range(args.numval)]
elif args.numval==1:
  try:
    data_val = [np.loadtxt(os.path.join(args.datadir,'val_0.txt'),
                          dtype=np.float32, delimiter=',', ndmin=2),]
  except FileNotFoundError:
    data_val = [np.loadtxt(os.path.join(args.datadir,'val.txt'),
                           dtype=np.float32, delimiter=',', ndmin=2),]

# -- Preprocessing
if args.preprocessing:
  train_absmax = np.absolute(data_train_all).max(axis=0)
else:
  train_absmax = np.ones((1,data_train_all.shape[1]), dtype=np.float32)
for i in range(args.numtrain):
  data_train[i] = data_train[i] / train_absmax

data_train_all = np.concatenate(data_train, axis=0)

if args.numval>0:
  for i in range(args.numval):
    data_val[i] = data_val[i] / train_absmax

# -- Some default settings
args.dimy = data_train[0].shape[1]
args.dimemb = args.dimemb or args.dimy*args.delay
args.dimobs = args.dimobs or args.dimemb

if args.add_learning_rate is None:
  args.add_learning_rate = args.learning_rate

# -- Set optimizer
if args.fixed_rate:
  use_optimizer = optimizers.SGD
else:
  use_optimizer = optimizers.SMORMS3

# -- Show settings
print('Learning Koopman Invariant Subspace')
print()
print('experiment name       : %s' % args.name)
for i in range(args.numtrain):
  print('  --> training data size   = (%d, %d)' %
        (data_train[i].shape[0], data_train[i].shape[1]))
if args.numval>0:
  for i in range(args.numval):
    print('  --> validation data size   = (%d, %d)' %
          (data_val[i].shape[0], data_val[i].shape[1]))
print('output directory      : %s' % args.outputdir)
print('gpu #                 : %d' % args.gpu)
print('minibatch size        : %d' % args.batchsize)
print('max. number of epochs : %d' % args.epoch)
print('delay length          : %d' % args.delay)
print('embedding dimension   : %d' % args.dimemb)
print('observable dimension  : %d' % args.dimobs)
print('alpha                 : %f' % args.alpha)
print('beta                  : %f' % args.beta)
print('gamma                 : %f' % args.gamma)
print('learning rate         : %f' % args.learning_rate)
if args.snapshot_interval>0:
  print('snapshot interval     : %d' % args.snapshot_interval)
if args.fixed_embedder:
  print('embedder is fixed     : -')
print()

# -- Set models and a loss function
loss = lkis.Loss(lkis.Embedder(args.dimy, args.delay, args.dimemb),
                 lkis.Network(args.dimemb, args.dimobs, args.dimy), args.alpha)
if args.fixed_embedder and args.dimy*args.delay==args.dimemb:
  loss.phi.l1.W = np.eye(args.dimemb, dtype=np.float32)
  loss.phi.l1.b = np.zeros(loss.phi.l1.b.shape, dtype=np.float32)

# -- GPU Setting
if args.gpu >= 0:
  import cupy
  cuda.get_device(args.gpu).use()
  loss.to_gpu()
  for i in range(args.numtrain):
    data_train[i] = cuda.to_gpu(data_train[i])
  if args.numval>0:
    for i in range(args.numval):
      data_val[i] = cuda.to_gpu(data_val[i])
  xp = cupy
else:
  xp = np

# -- Make datasets
dataset_train = lkis.DelayPairDataset(data_train, args.delay)
if args.numval>0:
  dataset_val = lkis.DelayPairDataset(data_val, args.delay)

# -- Set iterators
trs = True
if args.batchsize<1:
  args.batchsize = len(dataset_train)
  trs = False
train_iter = iterators.SerialIterator(dataset_train,
                                      batch_size=args.batchsize, shuffle=trs)
if args.numval>0:
  val_iter = iterators.SerialIterator(dataset_val,
                                      batch_size=len(dataset_val),
                                      repeat=False, shuffle=False)

# -- Set optimizers
optimizer1 = use_optimizer(lr=args.learning_rate); optimizer1.setup(loss.phi)
optimizer2 = use_optimizer(lr=args.learning_rate); optimizer2.setup(loss.net)
optimizer1.add_hook(optimizer_module.Lasso(args.beta))
optimizer2.add_hook(optimizer_module.WeightDecay(args.gamma))

# -- Set a trigger
if args.log_in_iteration:
  trigger = (1, 'iteration')
else:
  trigger = (1, 'epoch')

# -- Set a trainer
if args.fixed_embedder:
  optimizer_dict = {'net':optimizer2}
else:
  optimizer_dict = {'phi':optimizer1, 'net':optimizer2}
updater = lkis.Updater(train_iter, optimizer_dict, device=args.gpu, loss_func=loss)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outputdir)

# -- Add trainer extensions
trainer.extend(extensions.LogReport(trigger=trigger))
trainer.extend(extensions.ProgressBar())
#trainer.extend(extensions.dump_graph('loss'))

if args.numval>0:
  trainer.extend(lkis.Evaluator(val_iter, loss, device=args.gpu, trigger=trigger))
  trainer.extend(extensions.PrintReport([
    'epoch',
    'net/loss', 'validation/main/net/loss',
    'net/loss_kpm', 'validation/main/net/loss_kpm',
    'net/loss_rec', 'validation/main/net/loss_rec']))
else:
  trainer.extend(extensions.PrintReport([
    'epoch',
    'net/loss',
    'net/loss_kpm',
    'net/loss_rec']))

if args.snapshot_interval>0:
  snapshot_interval = (args.snapshot_interval, 'epoch')
  trainer.extend(extensions.snapshot(filename='trainer_epoch_{.updater.epoch}'),
                 trigger=snapshot_interval)

# -- Resume
if args.resume:
  serializers.load_npz(os.path.join(args.outputdir,'trainer_epoch_%s' % args.resume), trainer)
  print('-- resume from %s --' % args.resume)

# -- Run training
if args.epoch>0:
  start_time = time.time()
  print()
  trainer.run()
  print()
  elapsed_time = time.time() - start_time
  print('(%s) finished training for %d epochs with %f [sec]' % (args.name, args.epoch, elapsed_time))

#-- Additional training only for reconstructor, fixing the other parameters
if args.add_epoch>0:
  optimizer2 = use_optimizer(lr=0); optimizer2.setup(loss.net)
  optimizer3 = use_optimizer(lr=args.add_learning_rate); optimizer3.setup(loss.net.h)
  updater = lkis.Updater(train_iter, {'net':optimizer2, 'h':optimizer3}, device=args.gpu, loss_func=loss)
  trainer = training.Trainer(updater, (args.epoch+args.add_epoch, 'epoch'),
                             out=os.path.join(args.outputdir,'add'))
  trainer.extend(extensions.LogReport(trigger=trigger))
  if args.numval>0:
    trainer.extend(lkis.Evaluator(val_iter, loss, device=args.gpu, trigger=trigger))
    trainer.extend(extensions.PrintReport([
      'epoch',
      #'net/loss', 'validation/main/net/loss',
      #'net/loss_kpm', 'validation/main/net/loss_kpm',
      'net/loss_rec', 'validation/main/net/loss_rec']))
  else:
    trainer.extend(extensions.PrintReport([
      'epoch',
      #'net/loss',
      #'net/loss_kpm',
      'net/loss_rec']))
  print()
  trainer.run()
  print('(%s) finished additional training for %d epochs' % (args.name, args.add_epoch))
  print()

# -- Save the input arguments
with open(os.path.join(args.outputdir, 'args.json'), 'w') as f:
  args_d = vars(args)
  json.dump(args_d, f)

# -- Save the preprocessing quantities
np.save(os.path.join(args.outputdir, 'train_absmax'), train_absmax)
print('(%s) saved preprocesing quantities in %s' % (args.name, args.outputdir))

# -- Save the trained model
serializers.save_npz(os.path.join(args.outputdir, 'train_model_final.npz'), loss)
print('(%s) saved the trained model in %s' % (args.name, args.outputdir))

# -- Conduct DMD in g-space and save results
tmp = tuple(Variable(x, volatile='on')
            for x in dataset_module.concat_examples(dataset_train[:]))
g0, g1, h0, h1 = loss.net(*tmp, phi=loss.phi, train=False)
lam, w, z = lkis.dmd(g0.data, g1.data) # w,z are col-major
np.save(os.path.join(args.outputdir, 'train_lam'), lam)
np.save(os.path.join(args.outputdir, 'train_w'), w)
np.save(os.path.join(args.outputdir, 'train_z'), z)
print('(%s) saved DMD quantities in %s' % (args.name, args.outputdir))
# LKIS

This is a demo implementation of the following paper.

Naoya Takeishi, Yoshinobu Kawahara, and Takehisa Yairi, "Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition," in *Advances in Neural Information Processing Systems (Proc. of NIPS)*, vol. 30, pp. 1130-1140, 2017.

arXiv preprint: <https://arxiv.org/abs/1710.04340>

## Prerequisite

- python 3.5.2 or later
- numpy 1.12.1 or later
- scipy 0.19.0 or later
- chainer 1.23.0

## Files

* `lkis.py`
	- Core implementation of LKIS network.
* `train.py`
	- Script for training network.
* `predict.py`
	- Script for test by prediction based on a trained model.
* `exp_lorenz`
	- Root directory for experiment using Lorenz series. Dataset is included here.
* `matlab/`
	- MATLAB tools.


## Usage

```
python train.py [name] [options]
python predict.py [name] [options]
```

`[name]` specifies the name of the experiment.

### Example

```
python train.py lorenz --numval 1 --delay 7 --dimobs 5
python predict.py lorenz --save
```

The result can be inspected using `matlab/exp_lorenz.m`

## Important options

### train.py

* `--rootdir`
	- Root directory of an experiment. Data and results must be stored under this directory.
* `--datadir`
	- Name of the directory (under the root directory of the experiment) that contains datasets.
* `--outputdir`
	- Name of the directory (under the root directory of the experiment) where results will be stored.
* `--numtrain`
	- Number of training dataset files. If not specified, only one dataset file `train.txt` is used. If specified with 2, for example, `train_0.txt` and `train_1.txt` are used.
* `--numval`
	- Number of validation dataset files. If not specified, no validation dataset is used. If specified with 1, `val.txt` is used. If specified with 2, for example, `val_0.txt` and `val_1.txt` are used.
* `--delay`
	- Dimensionality of delay coordinates. $k$ in the paper.
* `--dimemb`
	- Dimensionality of the delay embedding. $p$ in the paper.
* `--dimobs`
	- Dimensionality of the learned observable. $n$ in the paper.
* `--epoch`
	- Number of epochs for SGD.

### predict.py

* `--numtest`
	- Similar to `--numval`.
* `--horizon`
	- Timestep horizon to which prediction is calculated.
* `--save`

## Author

*  **Naoya Takeishi** - [https://ntake.jp/](https://ntake.jp/)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

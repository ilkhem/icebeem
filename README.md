# Project structure
The project is set up as follows:
- all data generation is in the `data` folder. This includes synthetic ICA and IMCA data, and also code to read and load MNIST and other potential real datasets. Each type of data should have its own file. For instance, `imca.py`, `ica.py`, `mnist.py`, etc...
- all config files should be in the `config` folder.
- implementation of TCL should be in the `tcl` folder.
- different runners should go in the `runners` folder. For instance, we should have a runner for the nonlinear ICA experiment, one for the IMCA experiment, and one for the transfer learning on MNIST experiment.
- models and neural net implementations should go into the `models` folder. This should also include all flow implementation, potentially in a `flows` subfolder for tidiness.
- losses should be in the `losses` folder. These are FCE and DSM losses.
- metrics should be in the `metrics` folder. This includes the MCC and any other potential metrics.
- potentially add a `utils` folder for all utility files.

All folders should include an `__init__.py` so that python recognizes them as modules and we can call them from other files in the same root directory.
The `main.py` file is the only executable. It will run a runner from `runners` on a particular dataset in `data` according to a config file from `configs`.

Requirements:
 - pytorch 
 - tensorflow (for TCL)
 - tqdm

Tests:
- [x] TCL simulations (need to be tidied up and run through main.py)
- [x] iVAE simulations (need to be tidied up and run through main.py)
- [x] ICE-BeeM simulations (in progress)
- [x] MNIST experiments 
- [x] add CIFAR10
- [x] add fashionMNIST

### Simulations

To run simulations:
- `python run_simulations.py --dataset TCL --method ivae --config imca.yaml`
- `python run_simulations.py --dataset TCL --method tcl --config imca.yaml`
- `python run_simulations.py --dataset TCL --method icebeem --config imca.yaml`
and similarly for `-dataset IMCA`.

### Transfer learning
To run transfer learning experiments, we first we need to train both a conditional and unconditional (baseline) EBM on the classes
0-7:

```
python run_transfer.py --dataset MNIST --config mnist.yaml --doc mnistPreTrain
python run_transfer.py --dataset MNIST --config mnist.yaml --doc mnistUncondBaseline --baseline
```
Then, we fix the representation learnt by the feature extractor **f**, and train the secondary feature extractor **g** on
the unseen classes 8-9. We compare this to the baseline where we don't fix the feature extractor **f**. This is done by 
running
```
python run_transfer.py --dataset MNIST --transfer
```
Finally, to run the semi-supervised experiments:
```
python run_transfer.py --dataset MNIST --semisupervised
```

The same can be done on CIFAR-10 by changing the value of the flag `--dataset` to `CIFAR10` and of the flag 
`--config` to `cifar.yaml`.



# FINAL TESTS

### transfer learning:
```
python run_transfer.py --dataset MNIST --config mnist.yaml --doc mnistPreTrain
python run_transfer.py --dataset MNIST --config mnist.yaml --doc mnistPreTrain --transfer --all
python run_transfer.py --dataset MNIST --config mnist_baseline.yaml --all
```

### semi-supervised:
```
python run_transfer.py --dataset MNIST --config mnist.yaml --doc mnist --baseline
python run_transfer.py --dataset MNIST --config mnist.yaml --doc mnist --semisupervised
python run_transfer.py --dataset MNIST --config mnist.yaml --doc mnist --semisupervised --baseline
```
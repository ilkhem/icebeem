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
- [ ] ICE-BeeM simulations (in progress)
- [x] MNIST experiments 
- [x] add CIFAR10
- [x] add fashionMNIST

TODO (ilyes):
- [x] move all Datase objects to `data/`
- [x] remove legacy and unnecessary files
- [ ] work on FCE implementation and compare perf to Ricardo's
- [ ] add plotting option to `main.py`
- [ ] add weight checkpoints for easy plotting of Figures without training


to run simulations:
- `python3 main.py --dataset TCL --method ivae --config imca.yaml`
- `python3 main.py --dataset TCL --method tcl --config imca.yaml`
- `python3 main.py --dataset TCL --method icebeem --config imca.yaml`


to run MNIST experiments:

 - `python3 main.py --dataset MNIST --config mnist.yaml --doc mnistPreTrain`
 - `python3 main.py --dataset MNIST --config mnist.yaml --doc mnistUncondBaseline --unconditionalBaseline 1`

then run semi-supervised/transfer learning
 - `python3 run_analysis.py --dataset MNIST --run_semisupervised 1 `   or 
 - `python3 run_analysis.py --dataset MNIST --run_transfer 1 `


to run CIFAR10 experiemnts:

 - `python3 main.py --dataset CIFAR10 --config cifar.yaml --doc cifarPreTrain `
 - `python3 main.py --dataset CIFAR10 --config cifar.yaml --doc cifarUncondBaseline --unconditionalBaseline 1`

then run semi-supervised/transfer learning
 - `python3 run_analysis.py --dataset CIFAR10 --run_semisupervised 1 `   or
 - `python3 run_analysis.py --dataset CIFAR10 --run_transfer 1 `

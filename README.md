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



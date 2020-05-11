# UAI SUBMISSION 344

This repo is for the UAI submission number 344: ICE-BeeM: Identifiable Conditional Energy-Based Deep Models

This is the code to run the simulations presented in the manuscript as well as the transfer learning experiments (it also allows for additional experiments not discussed in the manuscript)

## Dependencies

This project was tested with the following versions:

- python 3.6 and 3.7
- pytorch 1.4
- tensorflow 1.14
- PyYAML 5.3.1
- seaborn 0.10
- scikit-learn 0.22.2
- scipy 1.4.1

## Running Simulations

We compared an ICE-BeeM model trained with flow contrastive estimation (FCE) to nonlinear ICA methods (iVAE and TCL).

 We first compared these methods on nonstationary data generated according to a nonlinear ICA model (we refer to this dataset as TCL. Second, the data was generated from a nonstationary IMCA model (where the latent variables are _dependent_) which we refer to as IMCA.

To reproduce the simulations, run for e.g.:

```
python simulations.py --dataset IMCA --method icebeem
```

Type `python simulations.py --help` to learn about the arguments:

```
usage: simulations.py [-h] [--dataset DATASET] [--method METHOD]
                      [--config CONFIG] [--run RUN] [--nSims NSIMS] [--test]
                      [--plot]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset to run experiments. Should be TCL or IMCA
  --method METHOD    Method to employ. Should be TCL, iVAE or ICE-BeeM
  --config CONFIG    Path to the config file
  --run RUN          Path for saving running related data.
  --nSims NSIMS      Number of simulations to run
  --test             Whether to evaluate the models from checkpoints
  --plot             Plot comparison of performances
```

The results of each simulation is saved in the value of the flag `--run` (defulat is `run/` folder).

## Running real data experiments

These experiments are run through the script `main.py`. Below are details on how to use the script. To learn about its arguments type `python main.py --help`:

```
usage: main.py [-h] [--config CONFIG] [--run RUN] [--nSims NSIMS]
               [--seed SEED] [--baseline] [--transfer] [--semisupervised]
               [--representation] [--subsetSize SUBSETSIZE] [--all] [--plot]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --run RUN             Path for saving running related data.
  --nSims NSIMS         Number of simulations to run
  --seed SEED           Random seed
  --baseline            Run the script for the baseline
  --transfer            Run the transfer learning experiments after pretraining
  --semisupervised      Run semi-supervised experiments
  --representation      Run CCA representation validation across multiple seeds
  --subsetSize SUBSETSIZE
                        Number of data points per class to consider -- only
                        relevant for transfer learning if not run with --all flag
  --all                 Run transfer learning experiment for many seeds and
                        subset sizes -- only relevant for transfer learning
  --plot                Plot selected experiment for the selected dataset
```

All options and choice of dataset are passed through a configuration file under the `configs` folder.
There are 4 main functions of interest:
- `train`: trains a (conditional) energy model on the dataset specified by the configuration file,
only considering the labels `0-n_labels`, the latter being specified by the config file.
- `transfer`: trains the secondary feature extractor **g** defining an ICE-BeeM on labels `n_labels-len(dset)`
while keeping the feature extractor **f** fixed (after it was trained using `train`).
- `semisupervised`: loads a feature extractor **f** from a (conditional) EBM pretrained on labels `0-n_labels` and uses it 
to classify classes `n_labels-len(dset)`
- `cca_representations`: trains a (conditional) energy model on a dataset (ignoring the `n_labels` field)
, saves the feature network **f**, and uses it to compute and save the learnt representation on the test split of
the dataset.

Below, we explain how to call these functions using the `main.py` script to recreate the experiments from the manuscript.
The experiments can be ran for:
- MNIST: by setting the flag `--config` to `mnist.yaml`.
- FashionMNIST: by setting the flag `--config` to `fashionmnist.yaml`.
- CIFAR10: by setting the flag `--config` to `cifar10.yaml`.
- CIFAR10: by setting the flag `--config` to `cifar100.yaml`.


### Transfer learning

In this experiment, we compare:

- Training an ICE-BeeM **f**.**g** on labels 0-7, then fixing **f** and learning only **g** on new unseen labels 8-9. 
- Training an ICE-BeeM **f**.**g** directly on labels 8-9.

The idea is to see whether the feature extractor **f** can learn meaningful representations from similar datasets, especially when the size of the dataset is small. We use the denoising dcore matching (DSM) loss as a comparison metric (lower is better). 

To run the experiment on MNIST:

```
# pretrain an ICE-BeeM on labels 0-7
python main.py --config mnist.yaml
# fix f and only learn g on labels 8-9 for many different dataset sizes and different seeds
python main.py --config mnist.yaml --transfer --all
# train an ICE-BeeM on labels 8-9 for many different dataset sizes and different seeds
python main.py --config mnist.yaml --transfer --baseline --all
```

The results are saved in the value of the flag `--run` (defulat is `run/` folder). To plot the comparison for MNIST after running the steps above:

```
python main.py --config mnist.yaml --transfer --plot
```

We also provide model checkpoints and experimental log to skip the training step.

### Semi-supervised learning

In the semi-supervised learning experiment, we compare:

- Training an ICE-BeeM **f**.**g** on labels 0-7, then using it to classify unseen labels 8-9.
- Training an unconditional EBM **h**.**1** on labels 0-7, then using it to classify unseen labels 8-9.

We use the classification accuracy as a comparison metric.

```
# pretrain an ICE-BeeM on labels 0-7 // same as first step in transfer learning exp
python main.py --config mnist.yaml
# pretrain an unconditional EBM on labels 0-7
python main.py --config mnist.yaml --baseline
# classify labels 8-9 using the pretrained ICE-BeeM
python main.py --config mnist.yaml --semisupervised
# classify labels 8-9 using the pretrained unconditional EBM
python main.py --config mnist.yaml --semisupervised --baseline
```

We also provide model checkpoints and experimental log to skip the training steps.


### Identifiability of representations

In these experiments we train multiple conditional and unconditional EBMs on various datasets and assess the identifiability of representations as discussed in Theorems 1-3. 

These experiments therefore do the following:
 - Train conditional and unconditional EBMs using different random initializations on the full train split of the dataset.
 - Study the learnt representations over held out test data. We compare the MCC over held out representations as well as MCC after linear transformation using CCA (this is akin to weak identifiability).

To run the experiment on MNIST:
```
# train an ICE-BeeM on all labels, for different seeds
python main.py --config mnist.yaml --nSims 10 --representation
# train an unconditional EBM on all labels, for different seeds
python main.py --config mnist.yaml --nSims 10 --representation --baseline
```

Then, MCC statistics can be computed and visualized using:

```
python main.py --config mnist.yaml --representation --plot
```

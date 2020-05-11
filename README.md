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

## Using SLURM
The bash script `slurm_main.sbatch` is a SLURM wrapper around `main.py` and allows to run multiple experiments in parallel
on a SLURM equipped server. 
You may have to change the `#SBATCH` configuration flags in the script according to your system.

This script sets `--nSims` to `1`, and allows the user to select the seeds for which to run the experiments using the
`--array` flag of `sbatch`. The rest of the arguments/flags can be passed as arguments of `slurm_main.sbatch`:
```
sbatch --array=some_seed_values slurm_main.sbatch --the_rest_of_the_arguments
``` 

### Examples

A use case is to run the transfer learning experiments in parallel:
```
# we use the --exclusive flag because the experiments ue 6-7Go of gpu memory
sbatch --exclusive --array=1-10 slurm_main.sbatch --config mnist.yaml --transfer --subsetSize 500
```
This is equivalent to running:
```
python main.py --config mnist.yaml --seed x --transfer --subsetSize 500
```
where `x` scans `[1-10]` for the value of the flag `--seed`. 
Following this approach requires to run the script for the flag `--subsetSize` in `[500, 1000, 2000, 3000, 4000, 5000, 6000]`.

___

Another use case for the identifiability of representations experiment is:
```
# we use the --exclusive flag because the experiments ue 6-7Go of gpu memory
sbatch --exclusive --array=42-51 slurm_main.sbatch --config mnist.yaml --representation
```
This is equivalent to:
```
python main.py --config mnist.yaml --seed 42 --nSims 10 --representation
```
with the added advantage that all seeds are run in parallel.

___

The script can also be used to run single seeds: if the `--array` flag is not set, then the seed value can be passed
as an argument `--seed` to `slurm_main.batch`:
```
sbatch --exclusive slurm_main.sbatch --config mnist.yaml --seed 42
```
is equivalent to
```
python main.py --config mnist.yaml --seed 42
```

___
To run everything, you can use the following bash script:
```
CONFIG_FILE=mnist.yaml
sbatch --exclusive slurm_main.sbatch --config $CONFIG_FILE 
for SIZE in 500 1000 2000 3000 4000 5000 6000
do
        sbatch --exclusive --array=0-4 slurm_main.sbatch --config $CONFIG_FILE --transfer --subsetSize $SIZE
        sbatch --exclusive --array=0-4 slurm_main.sbatch --config $CONFIG_FILE --transfer --subsetSize $SIZE --baseline
done
sbatch slurm_main.sbatch --config $CONFIG_FILE  --transfer --plot
sbatch --exclusive slurm_main.sbatch --config $CONFIG_FILE  --baseline
sbatch --exclusive slurm_main.sbatch --config $CONFIG_FILE  --semisupervised
sbatch --exclusive slurm_main.sbatch --config $CONFIG_FILE  --semisupervised --baseline
sbatch --exclusive --array=1-10 slurm_main.sbatch --config $CONFIG_FILE  --representation
sbatch --exclusive --array=1-10 slurm_main.sbatch --config $CONFIG_FILE  --representation --baseline
sbatch slurm_main.sbatch --config $CONFIG_FILE  --representation --plot
```


# ICE-BeeM: Identifiable Conditional Energy-Based Deep Models Based on Nonlinear ICA 

This repository contains code to run and reproduce the experiments presented in [ICE-BeeM: Identifiable Conditional Energy-Based Deep Models Based on Nonlinear ICA](https://arxiv.org/abs/2002.11537), published at NeurIPS 2020.
Work done by **Ilyes Khemakhem** (Gatsby Unit, UCL), **Ricardo Pio Monti** (Gatsby Unit, UCL), **Diederik P. Kingma** (Google Research) and **Aapo Hyvärinen** (University of Helsinki).

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

We compared an ICE-BeeM model trained with flow contrastive estimation ([FCE](https://arxiv.org/abs/1912.00589)) to nonlinear ICA methods ([iVAE](https://arxiv.org/abs/1907.04809) and [TCL](https://arxiv.org/abs/1605.06336)).

We first compared these methods on nonstationary data generated according to a nonlinear ICA model (we refer to this dataset as `TCL`. Second, the data was generated from a nonstationary IMCA model (where the latent variables are _dependent_) which we refer to as `IMCA`.

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

The result of each simulation is saved in the value of the flag `--run` (defulat is `run/` folder).

## Running real data experiments

These experiments are run through the script `main.py`. Below are details on how to use the script. To learn about its arguments type `python main.py --help`:

```
usage: main.py [-h] [--config CONFIG] [--run RUN] [--n-sims N_SIMS]
               [--seed SEED] [--baseline] [--transfer] [--semisupervised]
               [--representation] [--mcc] [--second-seed SECOND_SEED]
               [--subset-size SUBSET_SIZE] [--all] [--plot]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --run RUN             Path for saving running related data.
  --n-sims N_SIMS       Number of simulations to run
  --seed SEED           Random seed
  --baseline            Run the script for the baseline
  --transfer            Run the transfer learning experiments after
                        pretraining
  --semisupervised      Run semi-supervised experiments
  --representation      Run CCA representation validation across multiple
                        seeds
  --mcc                 compute MCCs -- only relevant for representation
                        experiments
  --second-seed SECOND_SEED
                        Second random seed for computing MCC -- only relevant
                        for representation experiments
  --subset-size SUBSET_SIZE
                        Number of data points per class to consider -- only
                        relevant for transfer learning if not run with --all
                        flag
  --all                 Run transfer learning experiment for many seeds and
                        subset sizes -- only relevant for transfer and
                        representation experiments
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
- `compute_representations`: trains a (conditional) energy model on a dataset (ignoring the `n_labels` field), saves the feature network **f**, and uses it to compute and save the learnt representation on the test split of
the dataset.

Below, we explain how to call these functions using the `main.py` script to recreate the experiments from the manuscript.
The experiments can be ran for:
- MNIST: by setting the flag `--config` to `mnist.yaml`.
- FashionMNIST: by setting the flag `--config` to `fashionmnist.yaml`.
- CIFAR10: by setting the flag `--config` to `cifar10.yaml`.
- CIFAR100: by setting the flag `--config` to `cifar100.yaml`.

The configuration files control most of the hyperparameters used for training and evaluation.
Notable fields are:
- `final_layer`: whether to apply a final FC layer to reduce the dimension of the latent space. To be used in conjunction with `feature_size`, which specifies the output (latent/feature) dimension.
- `architecture`: the architecture of the feature extractor **f**, can be `MLP`, `ConvMLP` or `Unet`. These different architectures are discussed in the Appendix.
- `positive`: whether to use positive features by applying a ReLU to the output of **f** (condition 3 of Theorem 2).
-  `augment`: whether to augment features by adding the square of **f** (condition 4 of Theorem 2).
When changing these fields in configuration `X.yaml`, make sure to also change them in `X_baseline.yaml`. The latter only serves for the baseline of transfer learning.

### Identifiability of representations

In these experiments we train multiple conditional and unconditional EBMs on various datasets and assess the identifiability of representations as discussed in Theorems 1-3.

These experiments therefore do the following:
 - Train conditional and unconditional EBMs using different random initializations on the full train split of the dataset.
 - Study the learnt representations over held out test data. We compare the MCC over held out representations as well as MCC after linear transformation using CCA (this is akin to weak identifiability).

To run the experiment on MNIST:
```

# train an ICE-BeeM on all labels, for seeds 0-9
# starting seed can be specified using --seed
python main.py --config mnist.yaml --representation --n-sims 10 
# train an unconditional EBM on all labels, for seeds 0-9
python main.py --config mnist.yaml --representation --n-sims 10 --baseline
```

Then, MCC statistics can be computed and visualized using:

```
# compute all pairwise MCCs for ICE-BeeM between seeds X and Y for X in {0..8} and Y in {X+1..9} 
# starting seed can be specified using --seed
python main.py --config mnist.yaml --representation --mcc --all --n-sims 10
# compute all pairwise MCCs for an unconditional EBM between seeds X and Y for X in {0..8} and Y in {X+1..9} 
python main.py --config mnist.yaml --representation --mcc --all --n-sims 10 --baseline
```

Finally, to visualize the MCC statistics with a boxplot:

```
# the number of seeds used for boxplot can be specified using --n-sims
# starting seed can be specified using --seed
python main.py --config mnist.yaml --representation --plot
```


### Transfer learning

In this experiment, we compare:

- Training an ICE-BeeM **f**.**g** on labels 0-7, then fixing **f** and learning only **g** on new unseen labels 8-9. This has the advantage of simplifying the learning, since **g** is much easier to train than **f** (we set it to a vector in our experiments).
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
# classify labels 8-9 using the pretrained ICE-BeeM -- outputs accuracy to stdout
python main.py --config mnist.yaml --semisupervised
# classify labels 8-9 using the pretrained unconditional EBM -- outputs accuracy to stdout
python main.py --config mnist.yaml --semisupervised --baseline
```


## Using SLURM
The bash scripts `slurm_main.sbatch` (GPU) and `slurm_main_cpu.sbatch` (CPU) are SLURM wrapper around `main.py` and allow to run multiple experiments in parallel
on a SLURM equipped server.
You may have to change the `#SBATCH` configuration flags in the scripts according to your system.

These scripts set `--n-sims` to `1`, and allows the user to select the seeds for which to run the experiments using the
`--array` flag of `sbatch`. The rest of the arguments/flags can be passed as arguments of `slurm_main.sbatch` or `slurm_main_cpu.sbatch`:
```
sbatch --array=some_seed_values slurm_main.sbatch --the_rest_of_the_arguments
```

### Examples

A use case is to run the transfer learning experiments in parallel:
```
sbatch --array=1-10 slurm_main.sbatch --config mnist.yaml --transfer --subset-size 500
```
This is equivalent to running:
```
python main.py --config mnist.yaml --seed x --transfer --subset-size 500
```
where `x` scans `[1-10]` for the value of the flag `--seed`.
Following this approach requires to run the script for the flag `--subset-size` in `[500, 1000, 2000, 3000, 4000, 5000, 6000]`.

___

Another use case for the identifiability of representations experiment is:
```
sbatch --array=42-51 slurm_main.sbatch --config mnist.yaml --representation
```
This is equivalent to:
```
python main.py --config mnist.yaml --seed 42 --n-sims 10 --representation
```
with the added advantage that all seeds are run in parallel.

___

The script can also be used to run single seeds: if the `--array` flag is not set, then the seed value can be passed
as an argument `--seed` to `slurm_main.batch`:
```
sbatch slurm_main.sbatch --config mnist.yaml --seed 42
```
is equivalent to
```
python main.py --config mnist.yaml --seed 42
```

### Run everything using SLURM
To maximize the use of parallel computing, we can get around the `--all` flag which launches loops inside the `main.py`
script by looping manually around the SLURM script as shown below. 
To run everything, you can use the following two step bash script:
```
CONFIG_FILE=mnist.yaml
sbatch slurm_main.sbatch --config $CONFIG_FILE
sbatch slurm_main.sbatch --config $CONFIG_FILE  --baseline
for SIZE in 0 500 1000 2000 3000 4000 5000 6000; do
        sbatch --array=0-4 slurm_main.sbatch --config $CONFIG_FILE --transfer --subset-size $SIZE --baseline
done
sbatch --array=0-19 slurm_main.sbatch --config $CONFIG_FILE  --representation
sbatch --array=0-19 slurm_main.sbatch --config $CONFIG_FILE  --representation --baseline
```
After these jobs finish and save the necessary weights and tensors, run:
```
CONFIG_FILE=mnist.yaml
typeset -i SEED SECSEED
for SEED in {0..18}; do
    for ((SECSEED=SEED+1;SECSEED<=19;SECSEED++)); do
        sbatch slurm_main_cpu.sbatch --config $CONFIG_FILE --representation --mcc --seed $SEED --second-seed $SECSEED
        sbatch slurm_main_cpu.sbatch --config $CONFIG_FILE --representation --mcc --seed $SEED --second-seed $SECSEED --baseline
    done
done
for SIZE in 0 500 1000 2000 3000 4000 5000 6000; do
        sbatch --array=0-4 slurm_main.sbatch --config $CONFIG_FILE --transfer --subset-size $SIZE
done
sbatch slurm_main.sbatch --config $CONFIG_FILE  --semisupervised
sbatch slurm_main.sbatch --config $CONFIG_FILE  --semisupervised --baseline
```

## References

If you find this code helpful/inspiring for your research, we would be grateful if you cite the following:


```bib
@inproceedings{khemakhem2020ice,
 author = {Khemakhem, Ilyes and Monti, Ricardo and Kingma, Diederik and Hyvarinen, Aapo},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {12768--12778},
 publisher = {Curran Associates, Inc.},
 title = {ICE-BeeM: Identifiable Conditional Energy-Based Deep Models Based on Nonlinear ICA},
 url = {https://proceedings.neurips.cc/paper/2020/file/962e56a8a0b0420d87272a682bfd1e53-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
  
  
## License
A full copy of the license can be found [here](LICENSE).

    Copyright (C) 2020 Ilyes Khemakhem

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.




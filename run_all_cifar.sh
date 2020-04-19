# run cifar exps

python run_transfer.py --dataset CIFAR10 --config cifar.yaml --doc cifar
python run_transfer.py --dataset CIFAR10 --config cifar.yaml --doc cifar --baseline
python run_transfer.py --dataset CIFAR10 --config cifar.yaml --doc cifar --semisupervised
python run_transfer.py --dataset CIFAR10 --config cifar.yaml --doc cifar --semisupervised --baseline
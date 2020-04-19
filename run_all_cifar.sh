# run cifar exps

python main.py --dataset CIFAR10 --config cifar.yaml --doc cifar
python main.py --dataset CIFAR10 --config cifar.yaml --doc cifar --baseline
python main.py --dataset CIFAR10 --config cifar.yaml --doc cifar --semisupervised
python main.py --dataset CIFAR10 --config cifar.yaml --doc cifar --semisupervised --baseline
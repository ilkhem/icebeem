# run representation learning exps

python3 main.py --dataset CIFAR10 --nSims 2 --config cifar.yaml --representation --retrainNets
python3 main.py --dataset CIFAR10 --nSims 2 --config cifar.yaml --representation --baseline --retrainNets


#python3 main.py --dataset MNIST --nSims 2 --config mnist.yaml --representation 
#python3 main.py --dataset MNIST --nSims 2 --config mnist.yaml --representation --baseline


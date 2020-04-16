#!/bin/bash
echo "Bash ..."

for n in 500 1000 2000 3000 4000 5000 6000
do
	for i in 1 2 3 4 5
	do 
		echo $n $i 
		CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset CIFAR10 --config cifar_baseline.yaml --doc cifarBaseline$n --SubsetSize $n --seed $i
	done
done


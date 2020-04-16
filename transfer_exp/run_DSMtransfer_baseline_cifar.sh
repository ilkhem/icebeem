#!/bin/bash
echo "Bash ..."

for n in 500 1000 2000 3000 4000 5000 600
do
	for i in 5
	do 
		echo $n $i 
		CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset CIFAR10 --config cifar_baseline.yaml --doc cifarBaseline$n --SubsetSize $n --seed $i
	done
done


#!/bin/bash
echo "Bash ..."

for n in 500 #500 1000 2000 3000 4000 5000 6000
do
	for i in 1 2 3 4 5 
	do 
		echo $n $i 
		CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset MNIST --config mnist_baseline.yaml --doc mnistBaseline --SubsetSize $n --seed $i
		# CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset MNIST --config mnist_baseline.yaml --doc mnistBaseline --SubsetSize $n --seed $i 
		# CUDA_VISIBLE_DEVICES=1 python3 main.py --runner BaselineRunner_Conditional --config transferBaseline.yml --doc TransferBaselineFolder --nSegments 2 --SubsetSize $n --RandomSeed $i 
	done
done


#!/bin/bash

# Feel free to use only parts of this file for testing...

RADIUS=5
PARTITION=gpu-short
ACCOUNT='nprg042s'
if [ -z "$ACCOUNT" ]; then
	echo "You need to set ACCOUNT variable in the script!"
	exit 1
fi

echo "----------"
echo "Compiling..."
srun -p $PARTITION -A $ACCOUNT make

echo "----------"
echo "Running serial version..."
srun -p $PARTITION -A $ACCOUNT --gres=gpu:1 ./cuda-blur-stencil serial $RADIUS ../data/lenna.pbm ../data/result-serial.pbm

echo "----------"
echo "Running CUDA version..."
srun -p $PARTITION -A $ACCOUNT --gres=gpu:1 ./cuda-blur-stencil cuda $RADIUS ../data/lenna.pbm ../data/result-cuda.pbm

echo "----------"
echo "Comaring results..."
if diff ../data/result-serial.pbm ../data/result-cuda.pbm; then
	echo "OK"
fi

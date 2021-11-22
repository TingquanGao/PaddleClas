#!/usr/bin/env bash

export FLAGS_START_PORT=7000
export CUDA_VISIBLE_DEVICES=0,1,2,3

datasets="cars cub sop"
for i in $datasets
do
    python -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./ppcls/configs/graduation/${i}.yaml
done

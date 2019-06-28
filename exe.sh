#!/bin/bash

HOME=/data1/home/cjh/DeepLearning/code/FusionNet
PROG=$HOME/seg-main.py
LOG=$HOME/result/log

list_batch_size='1'
epoch='100'
li='1e-4'
lf='1e-4' 
name="FusionNet"
mt='0.0'
wd='0.0'
iter='1'
moreInfo='original'
smoothing='off'

for bs in $list_batch_size; do
   CUDA_VISIBLE_DEVICES=$1 python -u $PROG --model_name $name --epochs $epoch --batch_size $bs --lr_initial $li --lr_final $lf \
   --iteration $iter --momentum $mt --weight_decay $wd --moreInfo $moreInfo --smoothing $smoothing
done
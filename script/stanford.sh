#!/bin/bash

HOME=/home/chojunghee/code/FusionNet
PROG=$HOME/main_stanford.py
LOG=$HOME/result/log

list_batch_size='2'
epoch='200'
li='1e-2'
lf='1e-2' 
#name="FusionNet"
name="Unet"
mt='0.9'
wd='0.0'
iter='1'
moreInfo='stanford,original'
smoothing='off'

for bs in $list_batch_size; do
   CUDA_VISIBLE_DEVICES=$1 python -u $PROG --model_name $name --epochs $epoch --batch_size $bs --lr_initial $li --lr_final $lf \
   --iteration $iter --momentum $mt --weight_decay $wd --moreInfo $moreInfo --smoothing $smoothing
done

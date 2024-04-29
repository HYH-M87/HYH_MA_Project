#!/usr/bin/env bash

IMGDIR=$1
ANNOTAIONDIR=$2
MODELWEIGHT=$3
MODELCFG=$4
SAMPLENUM=$5
LOGDIR=$6

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../code/predict_batch.py \
    $IMGDIR \
    $ANNOTAIONDIR \
    $MODELWEIGHT \
    $MODELCFG \
    $SAMPLENUM \
    $LOGDIR \

tensorboard  --logdir=$LOGDIR 

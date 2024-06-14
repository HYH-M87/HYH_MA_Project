#!/usr/bin/env bash
DATATYPE=$1
DATASET=$2
MODELWEIGHT=$3
MODELCFG=$4
SAMPLENUM=$5
LOGDIR=$6

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../code/predict_batch.py \
    $DATATYPE \
    $DATASET \
    $MODELWEIGHT \
    $MODELCFG \
    $SAMPLENUM \
    $LOGDIR \

tensorboard  --logdir=$LOGDIR --samples_per_plugin images=1000  --samples_per_plugin text=1000

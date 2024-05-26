#!/usr/bin/env bash
GONFIG=$1
CHECKPOINT=$2
SHOW=${3:- ''}
SHOWDIR=${4:- ''}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../../../4_dependencies/libraries/mmdetection/tools/test.py \
    $GONFIG \
    $CHECKPOINT \
    $SHOW \
    $SHOWDIR \



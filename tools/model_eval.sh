#!/usr/bin/env bash

MODE=$1
shift

# 根据 MODE 的值决定传入的 pred 参数数量
PRED=()
if [ "$MODE" == "model" ]; then
    PRED+=("$1")
    shift
    PRED+=("$1")
    shift
else
    PRED+=("$1")
    shift
fi

# 捕获剩余的参数
DATADIR=$1
PATCH_SIZE_H=$2
PATCH_SIZE_W=$3
OVERLAP_RATE=$4
SPLIT=$5
SCORE=$6
IOU=$7
SAVEDIR=$8
DESCRIPE=${9:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/model_eval/model_eval.py \
    $MODE \
    ${PRED[@]} \
    $DATADIR\
    $PATCH_SIZE_H\
    $PATCH_SIZE_W\
    $OVERLAP_RATE\
    $SPLIT\
    $SCORE\
    $IOU\
    $SAVEDIR\
    --descripe $DESCRIPE\

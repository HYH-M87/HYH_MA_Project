#!/usr/bin/env bash

# 读取配置文件
CONFIG_FILE=$1

MODE=$(jq -r '.MODE' $CONFIG_FILE)
PRED=$(jq -r '.PRED[]' $CONFIG_FILE)
DATADIR=$(jq -r '.DATADIR' $CONFIG_FILE)
PATCH_SIZE_H=$(jq -r '.PATCH_SIZE_H' $CONFIG_FILE)
PATCH_SIZE_W=$(jq -r '.PATCH_SIZE_W' $CONFIG_FILE)
OVERLAP_RATE=$(jq -r '.OVERLAP_RATE' $CONFIG_FILE)
SPLIT=$(jq -r '.SPLIT' $CONFIG_FILE)
SCORE=$(jq -r '.SCORE' $CONFIG_FILE)
IOU=$(jq -r '.IOU' $CONFIG_FILE)
SAVEDIR=$(jq -r '.SAVEDIR' $CONFIG_FILE)
DESCRIPE=$(jq -r '.DESCRIPE' $CONFIG_FILE)

# 将PRED参数转换为数组
PRED_ARR=($PRED)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/model_eval/model_eval.py \
  $MODE \
  "${PRED_ARR[@]}" \
  $DATADIR \
  $PATCH_SIZE_H \
  $PATCH_SIZE_W \
  $OVERLAP_RATE \
  $SPLIT \
  $SCORE \
  $IOU \
  $SAVEDIR \
  --descripe $DESCRIPE

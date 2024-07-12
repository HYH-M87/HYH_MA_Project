#!/usr/bin/env bash

# 读取配置文件
CONFIG_FILE=$1

DATADIR=$(jq -r '.DATADIR' $CONFIG_FILE)
JSONDIR=$(jq -r '.JSONDIR' $CONFIG_FILE)
PATCH_SIZE_H=$(jq -r '.PATCH_SIZE_H' $CONFIG_FILE)
PATCH_SIZE_W=$(jq -r '.PATCH_SIZE_W' $CONFIG_FILE)
OVERLAP_RATE=$(jq -r '.OVERLAP_RATE' $CONFIG_FILE)
SPLIT=$(jq -r '.SPLIT' $CONFIG_FILE)
SAVEDIR=$(jq -r '.SAVEDIR' $CONFIG_FILE)
DESCRIPE=$(jq -r '.DESCRIPE' $CONFIG_FILE)


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/hard_sample/hard_sample_analysis.py \
  $DATADIR \
  $JSONDIR \
  $PATCH_SIZE_H \
  $PATCH_SIZE_W \
  $OVERLAP_RATE \
  $SPLIT \
  $SAVEDIR \
  --descripe $DESCRIPE

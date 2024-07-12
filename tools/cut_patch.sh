#!/usr/bin/env bash

# 读取配置文件
CONFIG_FILE=$1
ORI_DATA=$(jq -r '.ORI_DATA' $CONFIG_FILE)
PROCESSED_DATA=$(jq -r '.PROCESSED_DATA' $CONFIG_FILE)
PATCH_SIZE_H=$(jq -r '.PATCH_SIZE_H' $CONFIG_FILE)
PATCH_SIZE_W=$(jq -r '.PATCH_SIZE_W' $CONFIG_FILE)
OVERLAP_RATE=$(jq -r '.OVERLAP_RATE' $CONFIG_FILE)
SPLIT=$(jq -r '.SPLIT' $CONFIG_FILE)
DESCRIPE=$(jq -r '.DESCRIPE' $CONFIG_FILE)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/data_preprocess/cut_patch_overlap.py \
  $ORI_DATA \
  $PROCESSED_DATA \
  $PATCH_SIZE_H \
  $PATCH_SIZE_W \
  $OVERLAP_RATE \
  $SPLIT \
  --descripe $DESCRIPE

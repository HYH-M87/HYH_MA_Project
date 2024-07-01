#!/usr/bin/env bash

ORI_DATA=$1
PROCESSED_DATA=$2
PATCH_SIZE_H=$3
PATCH_SIZE_W=$4
OVERLAP_RATE=$5
SPLIT=${6:- -1}
DESCRIPE=${7:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/data_preprocess/cut_patch_overlap.py \
    $ORI_DATA \
    $PROCESSED_DATA \
    $PATCH_SIZE_H \
    $PATCH_SIZE_W \
    $OVERLAP_RATE \
    $SPLIT\
    --descripe $DESCRIPE\

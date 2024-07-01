#!/usr/bin/env bash

ORI_DATA=$1
PROCESSED_DATA=$2
PATCH_SIZE_H=$3
PATCH_SIZE_W=$4
OVERLAP_RATE=$5
TARGET_SIZE_H=$6
TARGET_SIZE_W=$7
EXTEND=$8
SPLIT=$9
DESCRIPE=${10:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/data_preprocess/merge_patch.py \
    $ORI_DATA \
    $PROCESSED_DATA \
    $PATCH_SIZE_H \
    $PATCH_SIZE_W \
    $OVERLAP_RATE \
    $TARGET_SIZE_H \
    $TARGET_SIZE_W \
    $EXTEND \
    $SPLIT\
    --descripe $DESCRIPE\


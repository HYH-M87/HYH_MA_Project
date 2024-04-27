#!/usr/bin/env bash

TYPE=$1
ORI_DATA=$2
PROCESSED_DATA=$3
PATCH_SIZE_H=$4
PATCH_SIZE_W=$5
OVERLAP_RATE=$6
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../code/cut_patch_overlap.py \
    $TYPE \
    $ORI_DATA \
    $PROCESSED_DATA \
    $PATCH_SIZE_H \
    $PATCH_SIZE_W \
    $OVERLAP_RATE \

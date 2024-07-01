#!/usr/bin/env bash

CONFIG=$1
RESUME_PATH=${2:-"auto"}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../dependencies/libraries/mmdetection/tools/train.py \
    $CONFIG \
    --resume "$RESUME_PATH"



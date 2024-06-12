#!/usr/bin/env bash
START_PATH=$1
MODE=${2:- "c"}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../../5_tests/unit_test/create_git_keepfile.py \
    --mode "$MODE" \
    --start "$START_PATH"



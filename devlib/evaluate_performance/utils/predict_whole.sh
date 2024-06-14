DATATYPE=$1
DATASET=$2
EXPDIR=$3
PATCH_SIZE_H=$4
PATCH_SIZE_W=$5
OVERLAP_RATE=$6
SAMPLENUM=$7
SCORE_THRE=$8
IOU_THRE=$9
DESCRIPE=$10

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../code/predict_whole.py \
    $DATATYPE \
    $DATASET \
    $EXPDIR \
    $PATCH_SIZE_H \
    $PATCH_SIZE_W \
    $OVERLAP_RATE \
    $SAMPLENUM \
    $SCORE_THRE \
    $IOU_THRE \
    --descripe $DESCRIPE \
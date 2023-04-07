#!/bin/bash

LOW_SEED=1
HIGH_SEED=10
GATHER_DIR_ROOT="/checkpoint/plancaster/outputs/robohive_base/logs"
ENV_DIR="franka-FrankaPlanarPushReal_v2d"
EXP_DIR="planar_push_real_bc_100demos"
ROOT_DIR="$GATHER_DIR_ROOT/$ENV_DIR/$EXP_DIR"
MODEL_DIR="$ROOT_DIR/bc_models"

# Check that directory exists
if [ ! -d $ROOT_DIR ] 
then
    echo "$ROOT_DIR does not exist, exiting"
    exit 0
fi

# Make dst dir if it doesn't exist
if [ ! -d $MODEL_DIR ] 
then
    mkdir $MODEL_DIR
fi

for SEED in $(seq $LOW_SEED $HIGH_SEED)
do
    SRC_PATH="$ROOT_DIR/$SEED/models/0.pt"
    DST_PATH="$MODEL_DIR/bc_seed$SEED.pt" 

    # Check that model file exists
    if [ ! -f $SRC_PATH ]; 
    then
        echo "$SRC_PATH does not exist, skipping"
        continue  
    fi

    # Check that destination file doesn already exist
    if [ -f $DST_PATH ]; 
    then
        echo "$DST_PATH already exists, skipping"
        continue  
    fi
    
    cp $SRC_PATH $DST_PATH
    echo "Copied $SRC_PATH to $DST_PATH"
done
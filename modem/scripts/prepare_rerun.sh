#!/bin/bash

LOW_SEED=1
HIGH_SEED=6
LOG_STOP=5000
LOG_DELTA=2500
EPISODE_STOP=50
EPISODE_DELTA=1
ROOT_DIR="/checkpoint/plancaster/outputs/robohive_base"
ENV_NAME="franka-FrankaBinPick"
OLD_EXP="bin_pick_state-ensemble-uncertainty-weight"
NEW_EXP="bin_pick_state-ensemble-uncertainty-nosched"

OLD_LOG_DIR="$ROOT_DIR/logs/$ENV_NAME/$OLD_EXP"
OLD_EPISODE_DIR="$ROOT_DIR/episodes/$ENV_NAME/$OLD_EXP"
NEW_LOG_DIR="$ROOT_DIR/logs/$ENV_NAME/$NEW_EXP"
NEW_EPISODE_DIR="$ROOT_DIR/episodes/$ENV_NAME/$NEW_EXP"

# Check that directory exists
if [ ! -d $OLD_LOG_DIR ] 
then
    echo "$OLD_LOG_DIR does not exist, exiting"
    exit 0
fi

if [ ! -d $OLD_EPISODE_DIR ] 
then
    echo "$OLD_EPISODE_DIR does not exist, exiting"
    exit 0
fi

# Make sure dst dir doesn't already exist
if [ -d $NEW_LOG_DIR ] 
then
    echo "$NEW_LOG_DIR already exists, exiting"
    exit 0
fi

if [ -d $NEW_EPISODE_DIR ] 
then
    echo "$NEW_EPISODE_DIR already exists, exiting"
    exit 0
fi

for SEED in $(seq $LOW_SEED $HIGH_SEED)
do
    echo "Processing seed $SEED"

    OLD_MODEL_DIR="$OLD_LOG_DIR/$SEED/models/"
    NEW_MODEL_DIR="$NEW_LOG_DIR/$SEED/models/"
    OLD_ROLLOUT_DIR="$OLD_EPISODE_DIR/$SEED"
    NEW_ROLLOUT_DIR="$NEW_EPISODE_DIR/$SEED"

    if [ ! -d $OLD_MODEL_DIR ] 
    then
        echo "$OLD_MODEL_DIR does not exist, skipping"
        continue
    fi

    if [ ! -d $OLD_ROLLOUT_DIR ] 
    then
        echo "$OLD_ROLLOUT_DIR does not exist, skipping"
        continue
    fi

    if [ -d $NEW_MODEL_DIR ] 
    then
        echo "$NEW_MODEL_DIR already exists, skipping"
        continue
    fi

    if [ -d $NEW_ROLLOUT_DIR ] 
    then
        echo "$NEW_ROLLOUT_DIR already exists, skipping"
        continue
    fi

    echo "Copying models..."
    mkdir -p $NEW_MODEL_DIR
    for MODEL_CHKPT in $(seq 0 $LOG_DELTA $LOG_STOP)
    do
        cp "$OLD_MODEL_DIR/$MODEL_CHKPT.pt" "$NEW_MODEL_DIR/$MODEL_CHKPT.pt"
    done

    echo "Copying episodes..."
    mkdir -p $NEW_ROLLOUT_DIR
    for MODEL_CHKPT in $(seq -f "%010g" 0 $EPISODE_DELTA $EPISODE_STOP)
    do
        cp "$OLD_ROLLOUT_DIR/rollout$MODEL_CHKPT.pickle" "$NEW_ROLLOUT_DIR/rollout$MODEL_CHKPT.pickle"
    done
done

echo 'Done!'
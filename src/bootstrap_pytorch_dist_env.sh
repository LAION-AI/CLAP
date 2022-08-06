#!/bin/bash
# Set env vars for PyTorch
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}

export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=$head
export MASTER_PORT=29500 # default from torch launcher

export NUMBA_CACHE_DIR=/tmp/

export OMP_NUM_THREADS=7

export HOROVOD_CACHE_CAPACITY=0

wandb offline
export WANDB_DIR=/gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs_wandb
export WANDB_CONFIG_DIR=/gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs_wandb

echo "Setting env_var RANK=${RANK}"
echo "Setting env_var LOCAL_RANK=${LOCAL_RANK}"
echo "Setting env_var WORLD_SIZE=${WORLD_SIZE}"
echo "Setting env_var MASTER_ADDR=${MASTER_ADDR}"
echo "Setting env_var MASTER_PORT=${MASTER_PORT}"


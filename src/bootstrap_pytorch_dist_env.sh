#!/bin/bash
# Set env vars for PyTorch
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}

export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=$head
export MASTER_PORT=29500 # default from torch launcher

echo "Setting env_var RANK=${RANK}"
echo "Setting env_var LOCAL_RANK=${LOCAL_RANK}"
echo "Setting env_var WORLD_SIZE=${WORLD_SIZE}"
echo "Setting env_var MASTER_ADDR=${MASTER_ADDR}"
echo "Setting env_var MASTER_PORT=${MASTER_PORT}"


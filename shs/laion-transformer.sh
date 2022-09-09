#!/bin/bash
#SBATCH --comment clap
#SBATCH --partition=gpu
#SBATCH --job-name=mclap
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --exclude gpu-st-p4d-24xlarge-[66,106,141,144,225,324,347-350]

module load intelmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_PROTO=simple
export PATH=/opt/amazon/efa/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_ALGO=ring
export NCCL_DEBUG=info
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0


#export NCCL_P2P_DISABLE=1
#export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

source /home/tyz/clap/bin/activate
cd /home/tyz/CLAP/src

srun --comment clap --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --warmup 10000 \
    --batch-size=184 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=400 \
    --workers=10 \
    --use-bn-sync \
    --freeze-text \
    --amodel PANN-14 \
    --tmodel transformer \
    --report-to "wandb" \
    --wandb-notes "text-audio-freeze-text-lr-1e-3-8-dataset-model-pann-14" \
    --datasetnames "Clotho" "audiocaps" "BBCSoundEffects" "audioset" "free_to_use_sounds" "paramount_motion" "sonniss_game_effects" "wesoundeffects" \
    --datasetinfos "train" "unbalanced_train" "balanced_train" "valid" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --seed 3407 \
    --remotedata


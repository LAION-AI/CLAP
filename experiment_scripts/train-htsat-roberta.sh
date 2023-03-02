#!/bin/bash
#SBATCH --comment clap
#SBATCH --partition=g40423
#SBATCH --job-name=mclap
#SBATCH --nodes 3
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=6
#SBATCH --exclusive
#SBATCH --output=%x_%j.out

module load openmpi
module load cuda/11.7
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

source /fsx/yusong/clap/bin/activate
cd /fsx/yusong/CLAP/src
export TRANSFORMERS_CACHE=/fsx/yusong/transformers_cache

srun --comment clap --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --batch-size=96 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=6 \
    --use-bn-sync \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --warmup 3200 \
    --report-to "wandb" \
    --wandb-notes "10.16-clap-dataset-1#-htsat-roberta" \
    --datasetnames "Clotho" "audiocaps" \
    --datasetinfos "train" "unbalanced_train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --openai-model-cache-dir /fsx/yusong/transformers_cache \
    --logs /fsx/clap_logs \
    --seed 3407 \
    --remotedata \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained-audio /fsx/yusong/audio_pretrained_model/HTSAT-fullset-imagenet-map=0.467.ckpt

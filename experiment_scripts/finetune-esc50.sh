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

srun --comment clap --cpu_bind=v --accel-bind=gn python -m evaluate.eval_linear_probe \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --warmup 0 \
    --batch-size=160 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=100 \
    --workers=4 \
    --use-bn-sync \
    --freeze-text \
    --amodel PANN-14 \
    --tmodel roberta \
    --report-to "wandb" \
    --wandb-notes "10.14-finetune-esc50" \
    --datasetnames "esc50" \
    --datasetinfos "train" \
    --seed 3407 \
    --remotedata \
    --logs /fsx/clap_logs \
    --gather-with-grad \
    --lp-loss="ce" \
    --lp-metrics="acc" \
    --lp-lr=1e-4 \
    --lp-mlp \
    --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
    --openai-model-cache-dir /fsx/yusong/transformers_cache \
    --pretrained="/fsx/clap_logs/2022_10_14-04_05_14-model_PANN-14-lr_0.0001-b_160-j_6-p_fp32/checkpoints" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --optimizer "adam"
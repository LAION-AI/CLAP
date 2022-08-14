# set log path as the date and time
LOG="/gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs/clap_log_$(date +%Y%m%d_%H%M%S).log"
unset CUDA_VISIBLE_DEVICES

export NUMBA_CACHE_DIR=/tmp/

export OMP_NUM_THREADS=7

export HOROVOD_CACHE_CAPACITY=0

export WANDB_DIR=/gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs_wandb
export WANDB_CONFIG_DIR=/gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs_wandb
export WANDB_MODE=offline
wandb offline
# Export need to be strictly before than wandb offline

source ./bootstrap_pytorch_dist_env.sh

python -m training.main \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp16" \
    --pretrained="openai" \
    --warmup 10000 \
    --batch-size=96 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=400 \
    --workers=4 \
    --use-bn-sync \
    --freeze-text \
    --model HTSAT-tiny \
    --datasetnames "audiocaps" "BBCSoundEffects" "audioset" "free_to_use_sounds" "paramount_motion" "sonniss_game_effects" "wesoundeffects" \
    --datasetinfos "train" "unbalanced_train" "balanced_train" \
    --report-to "wandb" \
    --wandb-notes "text-audio-freeze-text-lr-1e-3-8-dataset-model" \
    --datasetpath /gpfs/alpine/scratch/wuyusong/csc499/clap_data/webdataset_tar \
    --logs /gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs \
    --openai-model-cache-dir /gpfs/alpine/scratch/wuyusong/csc499/clap_data/.cache/clip \
    --gather-with-grad \
    --local-loss \
    --no-eval

#2>&1 > $LOG
#    --report-to "wandb" \
#    --wandb-notes "text-audio-freeze-text-lr-1e-3-8-dataset-model" \

#     --datasetnames "audiocaps" "BBCSoundEffects" "audioset" "free_to_use_sounds" "paramount_motion" "sonniss_game_effects" "wesoundeffects" \
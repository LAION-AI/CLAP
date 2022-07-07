# set log path as the date and time
LOG="/gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs/clap_log_$(date +%Y%m%d_%H%M%S).log"
unset CUDA_VISIBLE_DEVICES

source ./bootstrap_pytorch_dist_env.sh

export NUMBA_CACHE_DIR='/tmp/'

python -m training.main \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --pretrained="openai" \
    --warmup 10000 \
    --batch-size=32 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=400 \
    --workers=2 \
    --use-bn-sync \
    --freeze-text \
    --model HTSAT-tiny \
    --resample-method="None" \
    --datasetnames "audiocaps" "audioset" \
    --datasetinfos "train" "unbalanced_train" "balanced_train" \
    --seed 3407 \
    --datasetpath /gpfs/alpine/scratch/wuyusong/csc499/clap_data/webdataset_tar \
    --logs /gpfs/alpine/scratch/wuyusong/csc499/clap_data/audio_clip_logs \
    --openai-model-cache-dir /gpfs/alpine/scratch/wuyusong/csc499/clap_data/.cache/clip \
    --horovod \
    --gather-with-grad

#2>&1 > $LOG
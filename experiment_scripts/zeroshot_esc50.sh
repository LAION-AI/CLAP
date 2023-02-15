# run from CLAP directory

python -m evaluate.eval_zeroshot_classification \
  --save-frequency 5 \
  --save-top-performance 3 \
  --save-most-recent \
  --dataset-type="webdataset" \
  --precision="fp32" \
  --warmup 0 \
  --batch-size=512 \
  --wd=0.0 \
  --epochs=50 \
  --workers=6 \
  --use-bn-sync \
  --freeze-text \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --report-to "wandb" \
  --wandb-notes "zeroshot-classification-esc50" \
  --datasetnames "esc50" \
  --datasetinfos "train" \
  --seed 3407 \
  --remotedata \
  --logs /fsx/clap_logs \
  --gather-with-grad \
  --openai-model-cache-dir /fsx/yusong/transformers_cache \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
  --pretrained="/fsx/clap_logs/2022_10_17-02_08_21-model_HTSAT-tiny-lr_0.0001-b_96-j_6-p_fp32/checkpoints"
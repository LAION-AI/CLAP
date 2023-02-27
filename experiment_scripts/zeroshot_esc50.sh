# run from CLAP directory
python -m evaluate.eval_zeroshot_classification \
  --dataset-type="webdataset" \
  --precision="fp32" \
  --batch-size=512 \
  --workers=6 \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --datasetnames "esc50_no_overlap" \
  --remotedata \
  --datasetinfos "train" \
  --seed 3407 \
  --logs ./logs \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --freeze-text \
  --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
  --pretrained="/fsx/clap_logs/2023_02_18-00_03_45-model_HTSAT-tiny-lr_0.0001-b_96-j_6-p_fp32/checkpoints"


python -m evaluate.eval_zeroshot_classification \
  --dataset-type="webdataset" \
  --precision="fp32" \
  --batch-size=512 \
  --workers=6 \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --report-to "wandb" \
  --wandb-notes "2.26-zeroshot-esc50-no-overlap-dataset-5#-fusion" \
  --datasetnames "esc50_no_overlap" \
  --remotedata \
  --datasetinfos "train" \
  --seed 3407 \
  --logs ./logs \
  --data-filling "repeatpad" \
  --data-truncating "fusion" \
  --enable-fusion \
  --fusion-type "aff_2d" \
  --freeze-text \
  --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
  --pretrained="/fsx/clap_logs/2023_02_26-01_28_30-model_HTSAT-tiny-lr_0.0001-b_96-j_6-p_fp32/checkpoints"


python -m evaluate.eval_zeroshot_classification \
  --dataset-type="webdataset" \
  --precision="fp32" \
  --batch-size=512 \
  --workers=4 \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --report-to "wandb" \
  --wandb-notes "2.26-zeroshot-esc50-no-overlap-dataset-2#-fusion" \
  --datasetnames "esc50_no_overlap" \
  --remotedata \
  --datasetinfos "train" \
  --seed 3407 \
  --logs ./logs \
  --data-filling "repeatpad" \
  --data-truncating "fusion" \
  --enable-fusion \
  --fusion-type "aff_2d" \
  --freeze-text \
  --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
  --pretrained="/fsx/clap_logs/2023_02_26-01_28_31-model_HTSAT-tiny-lr_0.0001-b_96-j_6-p_fp32/checkpoints"


# 90.3% accuracy
python -m evaluate.eval_zeroshot_classification \
  --dataset-type="webdataset" \
  --precision="fp32" \
  --batch-size=512 \
  --workers=6 \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --datasetnames "esc50_no_overlap" \
  --remotedata \
  --datasetinfos "train" \
  --seed 3407 \
  --logs ./logs \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --freeze-text \
  --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
  --pretrained="/fsx/clap_logs/2023_02_18-00_03_45-model_HTSAT-tiny-lr_0.0001-b_96-j_6-p_fp32/checkpoints/epoch_20.pt"
# run from CLAP directory

python -m evaluate.eval_zeroshot_classification \
  --dataset-type="webdataset" \
  --precision="fp32" \
  --batch-size=512 \
  --workers=6 \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --datasetnames "esc50" \
  --datasetpath "../" \
  --datasetinfos "train" \
  --seed 3407 \
  --logs ./logs \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
  --pretrained="../Model train on LAION-Audio-630K with fusion/checkpoints/epoch_top_0.pt"
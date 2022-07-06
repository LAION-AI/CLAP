# CLAP

Contrastive Language-Audio Pretraining, known as CLAP. Referring to the CLIP (Contrastive Language-Image Pretraining) architecture, similarly, the CLAP architecture is as follows.  
 <img width="906" alt="735af1fc4b786726348a421098e65b7" src="https://user-images.githubusercontent.com/53099276/176800216-9af8a6f2-ba06-45bf-b13e-ea4b83813218.png">

We adopt a CNN-based model --- PANN, and a transformer-based model --- HTS-AT, as our choices of audio encoder to encode the audio data, while loading and freezing the pre-trained text encoder from CLIP to encode the text data.

## About this project

This project is a project in [LAION](https://laion.ai/) that aims at learning better audio understanding and getting more audio data. This is a opensource project. We adopt the codebase of [open_clip](https://github.com/mlfoundations/open_clip) for this project. The major opensource contributers of this project are (in equal contribution): Yusong Wu, Tianyu Zhang, Ke Chen.

many thanks to <a href="https://github.com/cfoster0/CLAP">@cfoster0</a> for allowing us to use his repo name

## Work in Progress

This project is working in progress, thus the codebase and model might not be perfect or bug-free. We will very much appreciate any kind of contribution. If you would actively contribute to this project, please join the discord of LAION.

## Quick Start

```
conda create env -n clap python=3.10
conda activate clap
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
pip install -r requirements.txt
```
## Dataset format
We use training data in webdataset format. For details of our dataset please see https://github.com/LAION-AI/audio-dataset.

## Training
Firstly, please direct to the **src** folder:
```
cd src
```

To train on a single GPU machine, please run the following command. 
```
CUDA_VISIBLE_DEVICES=0 python -m training.main \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --pretrained="openai" \
    --warmup 10000 \
    --batch-size=184 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=400 \
    --workers=10 \
    --use-bn-sync \
    --freeze-text \
    --model HTSAT-tiny \
    --report-to "wandb" \
    --wandb-notes "text-audio-freeze-text-lr-1e-3-8-dataset-model-htsat-tiny" \
    --resample-method="None" \
    --datasetnames "Clotho" "audiocaps" "BBCSoundEffects" "audioset" "free_to_use_sounds" "paramount_motion" "sonniss_game_effects" "wesoundeffects" \
    --datasetinfos "train" "unbalanced_train" "balanced_train" \
    --datasetpath <webdataset_tar_path>
```

Here, ``model`` is set to choose the model from src/open_clip/model_configs/. Note that only 'PANN' and 'HTS-AT' models are supported, while others are image encoder models for previous CLIP.

And ``wandb-notes`` is set to report the log into the [wandb](https://github.com/wandb/client); ``datasetnames`` and ``datasetinfors`` are set to configure the dataset training and evalutaion specifications. Further information can be refered in ``src/training/params.py``

To train on a multi-GPU machine, please run the following command. Please feel checkout the help documentation in the argparse in ``params.py``.
```
torchrun --nproc_per_node 8  -m training.main \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --pretrained="openai" \
    --warmup 10000 \
    --batch-size=184 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=400 \
    --workers=10 \
    --use-bn-sync \
    --freeze-text \
    --model HTSAT-tiny \
    --report-to "wandb" \
    --wandb-notes "text-audio-freeze-text-lr-1e-3-8-dataset-model-htsat-tiny" \
    --resample-method="None" \
    --datasetnames "Clotho" "audiocaps" "BBCSoundEffects" "audioset" "free_to_use_sounds" "paramount_motion" "sonniss_game_effects" "wesoundeffects" \
    --datasetinfos "train" "unbalanced_train" "balanced_train" \
    --datasetpath <webdataset_tar_path>
```

## How to get audio feature?
Please refer to the ``evaluation()`` function in the ``train.py`` and `audio_infer()` function in `model.py`.

## How to get audio_text_rank?
Please refer to the ``get_metrics()`` function in the ``train.py``.

## Link to Pretrained Models
TODO


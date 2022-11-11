# CLAP

Contrastive Language-Audio Pretraining, known as CLAP. Referring to the CLIP (Contrastive Language-Image Pretraining) architecture, similarly, the CLAP architecture is as follows.  
<p align="center">
  <img src="./assets/audioclip-arch.png" alt="The Contrastive Language-Audio Pretraining Model Architecture" width="60%"/>
</p>



The repository contains code for the following paper:
 - [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/xxxx.xxxxx)

## About this project

This project is a project in [LAION](https://laion.ai/) that aims at learning better audio understanding and getting more audio data. 
This is an opensource project. We adopt the codebase of [open_clip](https://github.com/mlfoundations/open_clip) for this project. 
The major opensource contributers of this project are (in equal contribution): Yusong Wu, Tianyu Zhang, Ke Chen.

many thanks to <a href="https://github.com/cfoster0/CLAP">@cfoster0</a> for allowing us to use his repo name.

## Environment Installation
To install the same environment as we use, please run the following command:
```bash
conda create env -n clap python=3.10
conda activate clap
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
# you can also install pytorch by following the official instruction (https://pytorch.org/get-started/locally/)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
## Dataset format
We use training data in webdataset format. For details of our dataset please see https://github.com/LAION-AI/audio-dataset.

## Training, Fine-tuning and Evaluation
Please find the script of training, fine-tuning and evaluation (zero-shot and retrieval) in the [experiment_scripts](./experiment_scripts) folder. 
The scripts included there are the one we used to train our model on a SLURM cluster. 
You need to change the script to fit your own environment.
For example, in a single machine multi-GPU setting, you might want to use `torchrun` instead of `srun` to run the script.
To train on a single GPU machine, use `CUDA_VISIBLE_DEVICES=0 python -m ...` instead of `srun`.
We use [Weights and Biases](https://wandb.ai/site) for experiment logging. You need to configure the weights and biases in your environment.

## Loading Model and Inference
TODO

## Pretrained Models
The pretrained checkpoints can be found in [here](https://drive.google.com/drive/folders/1Ni8lZ2pryTESjgq8gELLQNM_HGdWtFrE?usp=sharing).
Please refer to the previous section for how to load and run the checkpoints.

The checkpoints list here for each model setting is the one with the highest average mAP score in training.
The average mAP score is calculated by averaging 4 scores: A-->T mAP@10 on AudioCaps, and T-->A mAP@10 on AudioCaps, A-->T mAP@10 on Clotho, and T-->A mAP@10 on Clotho.

## Citation
If you find this project and the LAION-Audio-630K dataset useful, please cite our paper:
```
TODO
```

## Acknowledgements

This project is working in progress, thus the codebase and model might not be perfect or bug-free. 
We will very much appreciate any kind of contribution or and issue raised.
If you find a bug or have any suggestion, please feel free to open an issue or contact us.
If you would actively contribute to this project, please join the discord of LAION.

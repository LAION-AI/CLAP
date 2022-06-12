import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from open_clip import create_model
from open_clip.utils import get_tar_path_from_dataset_name
from training.data import get_dataset_fn
from open_clip import tokenize
import glob
import json
import librosa
from tqdm import tqdm
import numpy as np


def get_output_from_single_audio(audio, text, model):
    audio_embedding = model.audio_infer(audio, hopsize=5 * 48000, key="embedding")
    audio_embedding = audio_embedding['embedding'].mean(dim=0, keepdim=True)
    audio_features = model.audio_projection(audio_embedding)
    audio_features = F.normalize(audio_features, dim=-1)
    text_features = model.encode_text(text)
    text_features = F.normalize(text_features, dim=-1)

    # CHANGE: before normalize or after
    audio_features_mlp = model.audio_transform(audio_features)
    text_features_mlp = model.text_transform(text_features)
    return audio_features, text_features, audio_features_mlp, text_features_mlp, model.logit_scale_a.exp(), model.logit_scale_t.exp()


def get_metrics(audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t):
    metrics = {}
    a_logits_per_audio = (logit_scale_a * audio_features @ text_features_mlp.t()).detach().cpu()
    a_logits_per_text = a_logits_per_audio.t().detach().cpu()
    t_logits_per_audio = (logit_scale_t * audio_features_mlp @ text_features.t()).detach().cpu()
    t_logits_per_text = t_logits_per_audio.t().detach().cpu()

    text_to_audio_logits = (a_logits_per_text + t_logits_per_text) / 2

    # repeat ground truth 5 times because Clotho has 5 text for 1 audio
    ground_truth = torch.repeat_interleave(torch.arange(len(text_features)//5), 5).view(-1, 1)

    ranking = torch.argsort(text_to_audio_logits, descending=True)
    preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
    preds = preds.detach().cpu().numpy()
    metrics[f"mean_rank"] = preds.mean() + 1
    metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k)
    # map@10
    metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))
    return metrics


if __name__ == '__main__':

    model_path = '/mnt/audio_clip/code/CLAP/src/logs/2022_06_05-17_53_56-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_0.pt'
    clotho_test_preprocessed_dir = "/mnt/audio_clip/processed_datasets/Clotho/test"

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='HTSAT-tiny')
    args = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = False
    pretrained = 'openai'

    device = torch.device(args.device)
    model, model_cfg = create_model(
        args.model,
        pretrained,
        precision='fp32',
        device=device,
        jit=False,
        force_quick_gelu=False,
    )

    # load model
    checkpoint = torch.load(model_path, map_location=device)
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith(
                "module"
        ):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # take every 5th file because clotho has 5 texts for 1 audio
    test_file_list = sorted(glob.glob(f"{clotho_test_preprocessed_dir}/*.flac"))[::5]

    audio_features_all = []
    text_features_all = []
    audio_features_mlp_all = []
    text_features_mlp_all = []
    logit_scale_a_all = []
    logit_scale_t_all = []

    with torch.no_grad():
        for file_path in tqdm(test_file_list):
            json_path = file_path.replace(".flac", ".json")
            with open(json_path, "r") as f:
                json_data = json.load(f)
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = torch.from_numpy(audio).to(device)
            text = json_data["original_data"]["all_captions"]
            text = tokenize(text).to(device)

            audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t = \
                get_output_from_single_audio(audio, text, model)

            audio_features_all.append(audio_features)
            text_features_all.append(text_features)
            audio_features_mlp_all.append(audio_features_mlp)
            text_features_mlp_all.append(text_features_mlp)
            logit_scale_a_all.append(logit_scale_a)
            logit_scale_t_all.append(logit_scale_t)

        metrics = get_metrics(
            audio_features=torch.cat(audio_features_all),
            text_features=torch.cat(text_features_all),
            audio_features_mlp=torch.cat(audio_features_mlp_all),
            text_features_mlp=torch.cat(text_features_mlp_all),
            logit_scale_a=logit_scale_a_all[0],
            logit_scale_t=logit_scale_t_all[0]
        )

        print(metrics)

    # TODO:[Tianyu]  load from DCASE contest dataset, get the retrieval file, and save csv file

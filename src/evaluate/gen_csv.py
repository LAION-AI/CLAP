import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from open_clip import create_model
from open_clip import tokenize
import glob
import json
import librosa
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


def get_output_from_single_audio(audio, text, model, device):
    audio_embedding = model.audio_infer(audio, hopsize=5 * 48000, key="embedding", device=device)[
        "embedding"
    ]
    if audio_embedding.ndim > 1:
        audio_embedding = audio_embedding.mean(dim=0, keepdim=True)
    audio_features = model.audio_projection(audio_embedding)
    audio_features = F.normalize(audio_features, dim=-1)
    text_features = model.encode_text(text)
    text_features = F.normalize(text_features, dim=-1)

    # CHANGE: before normalize or after
    audio_features_mlp = model.audio_transform(audio_features)
    text_features_mlp = model.text_transform(text_features)
    return (
        audio_features,
        text_features,
        audio_features_mlp,
        text_features_mlp,
        model.logit_scale_a.exp(),
        model.logit_scale_t.exp(),
    )


def get_audio_embeddings(audio, model, device):
    audio_embedding = model.audio_infer(audio, hopsize=5 * 48000, key="embedding", device=device)[
        "embedding"
    ]
    if audio_embedding.ndim > 1:
        audio_embedding = audio_embedding.mean(dim=0, keepdim=True)
    audio_features = model.audio_projection(audio_embedding)
    audio_features = F.normalize(audio_features, dim=-1)
    audio_features_mlp = model.audio_transform(audio_features)
    return audio_features, audio_features_mlp, model.logit_scale_a.exp()


def get_text_embeddings(text, model):
    text_features = model.encode_text(text)
    text_features = F.normalize(text_features, dim=-1)
    text_features_mlp = model.text_transform(text_features)
    return text_features, text_features_mlp, model.logit_scale_t.exp()


if __name__ == "__main__":
    file_dir = "/mnt/audio_clip/dcase/retrieval_audio/test/"
    audio_files = glob.glob(file_dir + "*")
    df = pd.read_csv("/mnt/audio_clip/dcase/retrieval_audio/retrieval_captions.csv")
    captions = list(df["caption"])

    model_path_all = [
        "/mnt/audio_clip/dcase_submission_systems/2022_05_07-19_16_07-model_HTSAT-tiny-lr_0.001-b_96-j_4-p_fp32/checkpoints/epoch_top_0.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_05_07-19_16_07-model_HTSAT-tiny-lr_0.001-b_96-j_4-p_fp32/checkpoints/epoch_top_1.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_05_07-19_16_07-model_HTSAT-tiny-lr_0.001-b_96-j_4-p_fp32/checkpoints/epoch_top_2.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_05-17_53_56-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_0.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_05-17_53_56-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_1.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_05-17_53_56-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_2.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_05_31-02_55_14-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_0.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_05_31-02_55_14-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_1.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_05_31-02_55_14-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_2.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_11-02_55_57-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_0.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_11-02_55_57-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_1.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_11-02_55_57-model_HTSAT-tiny-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_2.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_04-05_56_46-model_PANN-14-lr_0.001-b_184-j_6-p_fp32/checkpoints/epoch_top_0.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_04-05_56_46-model_PANN-14-lr_0.001-b_184-j_6-p_fp32/checkpoints/epoch_top_1.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_04-05_56_46-model_PANN-14-lr_0.001-b_184-j_6-p_fp32/checkpoints/epoch_top_2.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_11-21_01_57-model_PANN-14-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_0.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_11-21_01_57-model_PANN-14-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_1.pt",
        "/mnt/audio_clip/dcase_submission_systems/2022_06_11-21_01_57-model_PANN-14-lr_0.001-b_184-j_10-p_fp32/checkpoints/epoch_top_2.pt",
    ]
    model_type_all = [
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "HTSAT-tiny",
        "PANN-14",
        "PANN-14",
        "PANN-14",
        "PANN-14",
        "PANN-14",
        "PANN-14",
    ]

    clotho_test_preprocessed_dir = "/mnt/audio_clip/processed_datasets/Clotho/test"

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = False
    pretrained = "openai"

    audio_features_ensemble_all = []
    text_features_ensemble_all = []
    audio_features_mlp_ensemble_all = []
    text_features_mlp_ensemble_all = []
    logit_scale_a_ensemble_all = []
    logit_scale_t_ensemble_all = []
    # models = []
    for model_path, model_type in zip(model_path_all, model_type_all):
        device = torch.device(args.device)
        model, model_cfg = create_model(
            model_type,
            pretrained,
            precision="fp32",
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
            if next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            model.load_state_dict(sd)
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        audio_features_all = []
        text_features_all = []
        audio_features_mlp_all = []
        text_features_mlp_all = []
        logit_scale_a_all = []
        logit_scale_t_all = []

        with torch.no_grad():
            for file_path in audio_files:
                audio, sr = librosa.load(file_path, sr=48000, mono=True)
                audio = torch.from_numpy(audio).to(device)
                (
                    audio_features,
                    audio_features_mlp,
                    logit_scale_a,
                ) = get_audio_embeddings(audio, model,device)

                audio_features_all.append(audio_features)
                audio_features_mlp_all.append(audio_features_mlp)
                logit_scale_a_all.append(logit_scale_a)

            for text in captions:
                text = tokenize(text).to(device)
                text_features, text_features_mlp, logit_scale_t = get_text_embeddings(
                    text, model
                )
                text_features_all.append(text_features)
                text_features_mlp_all.append(text_features_mlp)
                logit_scale_t_all.append(logit_scale_t)

        audio_features_ensemble_all.append(torch.cat(audio_features_all))
        text_features_ensemble_all.append(torch.cat(text_features_all))
        audio_features_mlp_ensemble_all.append(torch.cat(audio_features_mlp_all))
        text_features_mlp_ensemble_all.append(torch.cat(text_features_mlp_all))
        logit_scale_a_ensemble_all.append(logit_scale_a_all[0])
        logit_scale_t_ensemble_all.append(logit_scale_t_all[0])

    text_to_audio_logits_ensemble_all = []
    for i in range(len(model_path_all)):
        audio_features = audio_features_ensemble_all[i]
        text_features = text_features_ensemble_all[i]
        audio_features_mlp = audio_features_mlp_ensemble_all[i]
        text_features_mlp = text_features_mlp_ensemble_all[i]
        logit_scale_a = logit_scale_a_ensemble_all[i]
        logit_scale_t = logit_scale_t_ensemble_all[i]

        a_logits_per_audio = (
            (logit_scale_a * audio_features @ text_features_mlp.t()).detach().cpu()
        )
        a_logits_per_text = a_logits_per_audio.t().detach().cpu()
        t_logits_per_audio = (
            (logit_scale_t * audio_features_mlp @ text_features.t()).detach().cpu()
        )
        t_logits_per_text = t_logits_per_audio.t().detach().cpu()

        text_to_audio_logits = (a_logits_per_text + t_logits_per_text) / 2
        text_to_audio_logits_ensemble_all.append(text_to_audio_logits)

    text_to_audio_logits_avg = sum(text_to_audio_logits_ensemble_all) / len(
        text_to_audio_logits_ensemble_all
    )
    ranks = torch.argsort(text_to_audio_logits_avg, dim=1, descending=True)
    results = []
    cols = ["caption"] + ["file_name_" + str(p + 1) for p in range(0, 10)]
    out = pd.DataFrame(columns=cols)
    for n in range(len(captions)):
        tmp = {"caption": captions[n]}
        tmp.update(
            {
                "file_name_" + str(p + 1): [os.path.basename(audio_files[ranks[n][p]])]
                for p in range(0, 10)
            }
        )
        entry = pd.DataFrame.from_dict(tmp)
        out = pd.concat([out, entry], ignore_index=True)
    out.to_csv("out.csv", index=False)

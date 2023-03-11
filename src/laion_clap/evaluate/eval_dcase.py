import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from open_clip import create_model
from open_clip import tokenize
import glob
import json
import librosa
from tqdm import tqdm
import numpy as np
import os
from laion_clap.training.params import parse_args


def get_output_from_single_audio(audio, text, model, device):

    # audio_embedding = model.audio_infer(audio, hopsize=5 * 48000, key="embedding", device=device)['embedding']
    # if audio_embedding.ndim > 1:
    #     audio_embedding = audio_embedding.mean(dim=0, keepdim=True)
    # else:
    #     audio_embedding = audio_embedding.unsqueeze(0)
    audio_features = model(audio, None, device)
    audio_features = F.normalize(audio_features, dim=-1)
    text_features = model(None, text, device=device)
    text_features = F.normalize(text_features, dim=-1)

    # CHANGE: before normalize or after
    audio_features_mlp = model.audio_transform(audio_features)
    text_features_mlp = model.text_transform(text_features)
    return audio_features, text_features, audio_features_mlp, text_features_mlp, model.logit_scale_a.exp(), model.logit_scale_t.exp()


def get_metrics(text_to_audio_logits):
    metrics = {}

    # repeat ground truth 5 times because Clotho has 5 text for 1 audio
    ground_truth = torch.repeat_interleave(torch.arange(len(text_features) // 5), 5).view(-1, 1)

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
    args = parse_args()

    model_path = args.pretrained

    clotho_test_preprocessed_dir = "/fsx/yusong/clotho_test_set/test"

    cudnn.benchmark = True
    cudnn.deterministic = False

    audio_features_ensemble_all = []
    text_features_ensemble_all = []
    audio_features_mlp_ensemble_all = []
    text_features_mlp_ensemble_all = []
    logit_scale_a_ensemble_all = []
    logit_scale_t_ensemble_all = []


    device = torch.device('cuda')
    model, clap_model_cfg = create_model(
        args.amodel,
        args.tmodel,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
        skip_params=False
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
    test_file_list = sorted(glob.glob(f"{clotho_test_preprocessed_dir}/*.flac"))

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
            audio, sr = librosa.load(file_path, sr=48000, mono=True)
            audio = torch.from_numpy(audio).to(device)
            audio = {'waveform': audio.unsqueeze(0), 'sample_rate': sr}
            text = json_data["text"]

            if args.tmodel == "transformer":
                from open_clip import tokenize
                text = tokenize(text)
            else:
                from laion_clap.training.data import tokenizer
                text = tokenizer(text, tmodel=args.tmodel)  # 5 texts for each audio

            audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t = \
                get_output_from_single_audio(audio, text, model, device)

            audio_features_all.append(audio_features.detach().cpu())
            text_features_all.append(text_features.detach().cpu())
            audio_features_mlp_all.append(audio_features_mlp.detach().cpu())
            text_features_mlp_all.append(text_features_mlp.detach().cpu())
            logit_scale_a_all.append(logit_scale_a.detach().cpu())
            logit_scale_t_all.append(logit_scale_t.detach().cpu())

    audio_features = torch.cat(audio_features_all)
    text_features = torch.cat(text_features_all)
    logit_scale_a = logit_scale_a_all[0]

    logits_per_audio = (logit_scale_a * audio_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_audio.t().detach().cpu()

    metrics = get_metrics(
        logits_per_text
    )

    print(metrics)

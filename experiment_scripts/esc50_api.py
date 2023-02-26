import laion_clap
from tqdm import tqdm
import glob
import json
import torch
import numpy as np

device = torch.device('cpu')  # ('cuda:0')

esc50_test_dir = './ESC50_1/test/'
class_index_dict_path = '../class_labels/ESC50_class_labels_indices_space.json'

# Load the model
model = laion_clap.CLAP_Module()
model.load_ckpt()
model.to(device)

# Get the class index dict
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

# Get all the data
audio_files = sorted(glob.glob(esc50_test_dir + '**/*.wav', recursive=True))
json_files = sorted(glob.glob(esc50_test_dir + '**/*.json', recursive=True))
ground_truth_idx = [class_index_dict[json.load(open(jf))['tag']] for jf in json_files]

with torch.no_grad():
    ground_truth = torch.tensor(ground_truth_idx).view(-1, 1)

    # Get text features
    all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]
    text_embed = model.get_text_embedding(all_texts).cpu()
    audio_embed = model.get_audio_embedding_from_filelist(x=audio_files).cpu()

    ranking = torch.argsort(audio_embed @ text_embed.t(), descending=True)
    preds = torch.where(ranking == ground_truth)[1]
    preds = preds.cpu().numpy()

    metrics = {}
    metrics[f"mean_rank"] = preds.mean() + 1
    metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k)
    # map@10
    metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

    print(
        f"Zeroshot Classification Results: "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

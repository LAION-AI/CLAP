# from training.data import *
# from easydict import EasyDict as edict
# from open_clip import create_model_and_transforms, trace_model, create_model
# from training.distributed import is_master, init_distributed_device, world_info_from_env
# from training.params import parse_args

# #     --save-frequency 1 \
# #     --zeroshot-frequency 1 \
# #     --dataset-type="toy" \
# #     --train-data="/mnt/audio_clip/test/data/indexes/balanced_train.h5"  \
# #     --val-data="/mnt/audio_clip/test/data/indexes/eval.h5"  \
# #     --train-ipc="/mnt/audio_clip/test/data/train_ipc.npy" \
# #     --val-ipc="/mnt/audio_clip/test/data/val_ipc.npy" \
# #     --csv-img-key filepath \
# #     --csv-caption-key title \
# #     --precision="fp32" \
# #     --warmup 10000 \
# #     --batch-size=32 \
# #     --lr=1e-3 \
# #     --wd=0.1 \
# #     --epochs=30 \
# #     --workers=8 \
# #     --model PANN-14
# args = parse_args()
# device = init_distributed_device(args)
# args.train_data = ["/mnt/audio_clip/webdataset_tar/audioset/unbalanced_train/0.tar", "/mnt/audio_clip/webdataset_tar/audioset/unbalanced_train/1.tar"]
# model, model_cfg = create_model(
#     args.model,
#     args.pretrained,
#     precision=args.precision,
#     device=device,
#     jit=args.torchscript,
#     force_quick_gelu=args.force_quick_gelu
# )

# data = get_wds_dataset(args, model_cfg, is_train=True)

# dl = data.dataloader
# for i, batch in enumerate(dl):
#     # print(i, batch)
#     # print("batch.keys", batch.keys())
#     # # 'hdf5_path', 'index_in_hdf5', 'audio_name', 'waveform', 'target', 'text'
#     # print("hdf5_path", batch["hdf5_path"])
#     # print("index_in_hdf5", batch["index_in_hdf5"])
#     # print("audio_name", batch["audio_name"])
#     # print("waveform", batch["waveform"])
#     # print("waveform.shape", batch["waveform"].shape)
#     # print("target", batch["target"])
#     # print("target.shape", batch["target"].shape)
#     # print("text", batch["text"])
#     # print("text.shape", batch["text"].shape)
#     print("osr", batch[-1])
#     print()
#     break

import webdataset as wds
import soundfile as sf
import io
import os
import random
import copy
from tqdm import tqdm
import s3fs
import numpy as np
import shutil
import json
from open_clip.utils import load_p
import logging
lp_class_label = load_p("audioset_class_labels_indices.pkl")

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

def preprocess(
    sample,
):
    """
    Preprocess a single sample for wdsdataloader.
    """
    keys = list(sample.keys())
    audio_ext = "flac"
    audio_data, orig_sr = sf.read(io.BytesIO(sample[audio_ext]))
    sample["waveform"] = audio_data
    json_dict_raw = json.loads(sample["json"].decode("utf-8"))
    sample["multi_class_label"] = np.zeros(len(lp_class_label))
    for x in json_dict_raw["class_names"]:
        sample["multi_class_label"][lp_class_label[x]] = 1
    return sample


# def check(j):
#     input_shards = ["/mnt/audio_clip/webdataset_tar/audioset/unbalanced_train/"+str(i)+".tar" for i in range(j,3735)]
#     # input_shards = ["/mnt/audio_clip/webdataset_tar/audioset/unbalanced_train/"+str(i)+".tar" for i in range(38,56)]
#     # input_shards = ["/mnt/audio_clip/webdataset_tar/audioset/eval/28.tar"]
#     pipeline = [wds.SimpleShardList(input_shards)]
#     _SHARD_SHUFFLE_SIZE = 2000
#     _SHARD_SHUFFLE_INITIAL = 500
#     _SAMPLE_SHUFFLE_SIZE = 5000
#     _SAMPLE_SHUFFLE_INITIAL = 1000
#     pipeline.extend([
#         # wds.detshuffle(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL),
#         wds.split_by_node,
#         wds.split_by_worker,
#         # at this point, we have an iterator over the shards assigned to each worker at each node
#         wds.tarfile_to_samples(handler=log_and_continue),
#         # wds.shuffle(
#         #     bufsize=_SAMPLE_SHUFFLE_SIZE,
#         #     initial=_SAMPLE_SHUFFLE_INITIAL,
#         #     rng=random.Random(1)),
#         #wds.repeatedly,  # FIXME determine if this is beneficial
#     ])
#     pipeline.extend([wds.map(preprocess),wds.to_tuple("__url__", "__key__", "waveform"),wds.batched(1)])
#     dataset = wds.DataPipeline(*pipeline)
#     dataloader = wds.WebLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
#     for i, batch in enumerate(dataloader):
#         previous_batch = copy.deepcopy(batch)
#     return previous_batch
# errors = []
# j = 279
# while True:
#     try:
#         previous_batch = check(j)
#     except:
#         print(int(previous_batch[0][0][0].split("/")[-1].split(".tar")[0])+1)
#         check(int(previous_batch[0][0][0].split("/")[-1].split(".tar")[0])+1)
#         errors.append(int(previous_batch[0][0][0].split("/")[-1].split(".tar")[0]))
#         pass
# print(errors)


# 假设有数据集dataset_all = [‘aaa’, ‘bbb’, ‘ccc’]和他们的split
# dataset_split = {
#     "audiocaps": ["train", "valid", "test"],
#     "audioset": ["balanced_train", "unbalanced_train", "eval"],
#     "BBCSoundEffects": ["train", "test"],
#     "Clotho": ["train", "test", "valid"],
# }
# for dataset_name in ["audiocaps", "audioset", "BBCSoundEffects", "Clotho"]:
#     for split in dataset_split[dataset_name]:
#         if not os.path.exists(f"./json_files/{dataset_name}/{split}"):
#             os.makedirs(f"./json_files/{dataset_name}/{split}")
#         os.system(
#             f"aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json ./json_files/{dataset_name}/{split}/sizes.json"
#         )

try:
    # input_shards = [
    #     f"pipe:aws s3 cp s3://s-laion-audio/webdataset_tar/audioset/unbalanced_train/{i}.tar -"
    #     for i in range(0, 3734)
    # ]
    input_shards = ["file://D:/eval.tar"]
    # input_shards = ["/mnt/audio_clip/webdataset_tar/audioset/eval/28.tar"]
    pipeline = [wds.SimpleShardList(input_shards)]
    _SHARD_SHUFFLE_SIZE = 2000
    _SHARD_SHUFFLE_INITIAL = 500
    _SAMPLE_SHUFFLE_SIZE = 5000
    _SAMPLE_SHUFFLE_INITIAL = 1000
    pipeline.extend(
        [
            # wds.detshuffle(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL),
            wds.split_by_node,
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker at each node
            wds.tarfile_to_samples(handler=log_and_continue),
            # wds.shuffle(
            #     bufsize=_SAMPLE_SHUFFLE_SIZE,
            #     initial=_SAMPLE_SHUFFLE_INITIAL,
            #     rng=random.Random(1)),
            # wds.repeatedly,  # FIXME determine if this is beneficial
        ]
    )
    pipeline.extend(
        [
            wds.map(preprocess),
            wds.to_tuple("__url__", "__key__", "waveform", "multi_class_label"),
            wds.batched(1),
        ]
    )
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    old_k = 0
    old_batch = None
    for k, batch in tqdm(enumerate(dataloader)):
        print(k)
        old_k = k
        old_batch = copy.deepcopy(batch)
        print(batch)
except:
    print(old_k)
    print(old_batch)
    pass
# for i in tqdm(reversed(range(1226 * 2, 1226 * 3 + 1))):
#     try:
#         input_shards = [
#             f"pipe:aws s3 cp s3://s-laion-audio/webdataset_tar/audiocaps/test/{i}.tar -",
#         ]
#         # input_shards = ["/mnt/audio_clip/webdataset_tar/audioset/eval/28.tar"]
#         pipeline = [wds.SimpleShardList(input_shards)]
#         _SHARD_SHUFFLE_SIZE = 2000
#         _SHARD_SHUFFLE_INITIAL = 500
#         _SAMPLE_SHUFFLE_SIZE = 5000
#         _SAMPLE_SHUFFLE_INITIAL = 1000
#         pipeline.extend(
#             [
#                 # wds.detshuffle(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL),
#                 wds.split_by_node,
#                 wds.split_by_worker,
#                 # at this point, we have an iterator over the shards assigned to each worker at each node
#                 wds.tarfile_to_samples(handler=log_and_continue),
#                 # wds.shuffle(
#                 #     bufsize=_SAMPLE_SHUFFLE_SIZE,
#                 #     initial=_SAMPLE_SHUFFLE_INITIAL,
#                 #     rng=random.Random(1)),
#                 # wds.repeatedly,  # FIXME determine if this is beneficial
#             ]
#         )
#         pipeline.extend(
#             [
#                 wds.map(preprocess),
#                 wds.to_tuple("__url__", "__key__", "waveform"),
#                 wds.batched(1),
#             ]
#         )
#         dataset = wds.DataPipeline(*pipeline)
#         dataloader = wds.WebLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#         for k, batch in enumerate(dataloader):
#             previous_batch = copy.deepcopy(batch)
#             print(batch)
#     except:
#         # print(i)
#         # print(previous_batch)
#         pass

# main training loop
# generator = iter(dataloader)
# for i in range(9999999999999):
#     try:
#         # Samples the batch
#         if i>0:
#             previous = copy.deepcopy(batch)
#         batch = next(generator)
#         print(batch)
#     except:
#         errors.append(previous)
#         pass
# print(errors)

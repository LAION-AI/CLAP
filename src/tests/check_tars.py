import webdataset as wds
import soundfile as sf
import io
import os
import random
import copy
from tqdm import tqdm
import shutil
import argparse
import traceback
import logging
import json
from laion_clap import tokenize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tar-path",
        type=str,
        default=None,
        help="Path to the tars",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="start from tar-path + start",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=99999,
        help="end with tar-path + end",
    )
    parser.add_argument(
        "--exclude",
        nargs='+',
        default=None,
        help="exclude tar-path + exclude",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--order",
        default=False,
        action='store_true',
        help="if keep the search order accendingly",
    )
    args = parser.parse_args()
    return args

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
    audio_ext = "flac"
    text_ext = "json"
    audio_data, orig_sr = sf.read(io.BytesIO(sample[audio_ext]))
    json_dict_raw = json.loads(sample[text_ext].decode("utf-8"))
    sample["waveform"] = audio_data
    texts = json_dict_raw["text"]
    if isinstance(texts, list) and isinstance(texts[0], str) and len(texts) > 1:
        texts = random.choice(texts)
    sample["raw_text"] = texts
    sample["text"] = tokenize(texts)
    return sample

if __name__ == "__main__":
    args = parse_args()
    tar_path = args.tar_path
    idx_list = list(range(args.start, args.end))
    if args.exclude != None:
        for x in args.exclude:
            idx_list.remove(x)
    if not args.order:
        random.shuffle(idx_list)
    if "aws" in tar_path:
        args.local = False
    if args.local:
        input_shards = [os.path.join(args.tar_path, str(i)+".tar") for i in idx_list]
    else:
        input_shards = [os.path.join(args.tar_path, str(i)+".tar -") for i in idx_list]
    pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend(
        [
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.map(preprocess),
            wds.to_tuple("__url__", "__key__", "waveform"),
            wds.batched(1),
        ]
    )
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    old_k = 0
    old_batch = None
    try:
        for k, batch in tqdm(enumerate(dataloader)):
            print("k:", k)
            print("batch:", batch)
            old_k = k
            old_batch = copy.deepcopy(batch)
    except:
        with open("check_tar_log.txt","a") as file:
            traceback.print_exc(file = file)
        print("old_k:", old_k)
        print("old_batch:", old_batch)
        pass

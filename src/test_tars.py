import webdataset as wds
import soundfile as sf
import io
import os
import random
import copy
from tqdm import tqdm
import s3fs
import shutil
import argparse

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
        default=32,
    )
    parser.add_argument(
        "--order",
        default=False,
        action='store_true',
        help="if keep the search order accendingly",
    )
    args = parser.parse_args()
    return args


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
    return sample

if __name__ == "__main__":
    try:
        args = parse_args()
        tar_path = args.tar_path
        idx_list = list(range(args.start, args.end))
        for x in args.exclude:
            idx_list.remove(x)
        if not args.order:
            random.shuffle(idx_list)
        input_shards = [os.path.join(args.tar_path, str(i)+".tar")) for i in idx_list]
        pipeline = [wds.SimpleShardList(input_shards)]
        pipeline.extend(
            [
                wds.map(preprocess),
                wds.to_tuple("__url__", "__key__", "waveform"),
                wds.batched(1),
            ]
        )
        dataset = wds.DataPipeline(*pipeline)
        dataloader = wds.WebLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        old_k = 0
        old_batch = None
        for k, batch in tqdm(enumerate(dataloader)):
            old_k = k
            old_batch = copy.deepcopy(batch)
    except:
        print(old_k)
        print(old_batch)
        pass
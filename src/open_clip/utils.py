import numpy as np
import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import logging
import h5py
from tqdm import tqdm
import random


def freeze_batch_norm_2d(module, module_match={}, name=""):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(
        module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


def exist(dataset_name, dataset_type):
    """
    Check if dataset exists
    """
    if dataset_name in ["Clotho", "audiocaps"] and dataset_type in [
        "train",
        "test",
        "valid",
    ]:
        return True
    elif dataset_name == "BBCSoundEffects" and dataset_type in ["train", "test"]:
        return True
    elif dataset_name == "audioset" and dataset_type in [
        "balanced_train",
        "unbalanced_train",
        "eval",
    ]:
        return True
    else:
        return False


def get_tar_path_from_dataset_name(
    dataset_names,
    dataset_types,
    islocal,
    template="/mnt/audio_clip/code/CLAP/src/data/datasetname/datasettype.txt",
    proportion=1,
):
    """
    Get tar path from dataset name and type
    """
    txt_paths = []
    for i in range(len(dataset_names)):
        for j in range(len(dataset_types)):
            if exist(dataset_names[i], dataset_types[j]):
                txt_loc = template.replace("datasetname", dataset_names[i]).replace(
                    "datasettype", dataset_types[j]
                )
                txt_paths.append(txt_loc)
            else:
                print(
                    "Skipping dataset " + dataset_names[i] + " with " + dataset_types[j]
                )
                continue
    return get_tar_path_from_txts(txt_paths, islocal=islocal, proportion=proportion)


def get_tar_path_from_txts(txt_path, islocal, proportion=1):
    """
    Get tar path from txt path
    """
    if isinstance(txt_path, (list, tuple)):
        return sum(
            [
                get_tar_path_from_txts(
                    txt_path[i], islocal=islocal, proportion=proportion
                )
                for i in range(len(txt_path))
            ],
            [],
        )
    if isinstance(txt_path, str):
        with open(txt_path) as f:
            lines = f.readlines()
        if islocal:
            lines = [
                lines[i]
                .split("\n")[0]
                .replace("pipe:s3cmd get s3://laion-audio/", "/mnt/audio_clip/")
                for i in range(len(lines))
            ]
        else:
            lines = [lines[i].split("\n")[0] for i in range(len(lines))]
        if proportion != 1:
            print("Sampling tars with proportion of {}".format(proportion))
            lines = random.sample(lines, int(proportion * len(lines)))
        return lines


def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [
        np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)
    ]
    return np.array(mixup_lambdas).astype(np.float32)


def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)
    Returns:
      out: (batch_size, ...)
    """
    out = (
        x.transpose(0, -1) * mixup_lambda
        + torch.flip(x, dims=[0]).transpose(0, -1) * (1 - mixup_lambda)
    ).transpose(0, -1)
    return out


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""


def process_ipc(index_path, classes_num, filename):
    # load data
    logging.info("Load Data...............")
    ipc = [[] for _ in range(classes_num)]
    with h5py.File(index_path, "r") as f:
        for i in tqdm(range(len(f["target"]))):
            t_class = np.where(f["target"][i])[0]
            for t in t_class:
                ipc[t].append(i)
    print(ipc)
    np.save(filename, ipc)
    logging.info("Load Data Succeed...............")


def save_to_dict(s, o_={}):
    sp = s.split(": ")
    o_.update({sp[0]: float(sp[1])})
    return o_


def get_data_from_log(txt_path):
    """
    Output dictionary from out.txt log file
    """
    with open(txt_path) as f:
        lines = f.readlines()
    val_data = {}
    train_data = {}
    train_losses = []
    train_losses_epoch = []
    for i in range(len(lines)):
        if "| INFO |" in lines[i]:
            if "Eval Epoch" in lines[i]:
                if "val_loss" in lines[i]:
                    # float(regex.sub("", lines[310].split("	")[-1]).replace(" ", ""))
                    line = lines[i].split("Eval Epoch: ")[-1]
                    num_epoch = int(line.split("	")[0].split(" ")[0])
                    d = {
                        line.split("	")[0]
                        .split(" ")[1]
                        .replace(":", ""): float(line.split("	")[0].split(" ")[-1])
                    }
                    for i in range(1, len(line.split("	"))):
                        d = save_to_dict(line.split("	")[i], d)
                    val_data[num_epoch] = d
            elif "Train Epoch" in lines[i]:
                num_epoch = int(lines[i].split("Train Epoch: ")[1][0])
                loss = float(lines[i].split("Loss: ")[-1].split(" (")[0])
                train_losses.append(loss)
                train_losses_epoch.append(num_epoch)
    for i in range(len(train_losses)):
        train_data[i] = {
            "num_epoch": train_losses_epoch[i],
            "train_loss": train_losses[i],
        }
    return train_data, val_data

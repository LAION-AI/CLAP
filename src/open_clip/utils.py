from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module, module_match={}, name=''):
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
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
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
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


def exist(dataset_name, dataset_type):
    """
    Check if dataset exists
    """
    if dataset_name in ["Clotho", "audiocaps"] and dataset_type in ["train", "test", "valid"]:
        return True
    elif dataset_name == "BBCSoundEffects" and dataset_type in ["train", "test"]:
        return True
    elif dataset_name == "audioset" and dataset_type in ["balanced_train", "unbalanced_train", "eval"]:
        return True
    else:
        return False
        

def get_tar_path_from_dataset_name(dataset_names, dataset_types, islocal, template='/mnt/audio_clip/code/CLAP/src/data/datasetname/datasettype.txt'):
    """
    Get tar path from dataset name and type
    """
    txt_paths = []
    for i in range(len(dataset_names)):
        for j in range(len(dataset_types)):
            if exist(dataset_names[i],dataset_types[j]):
                txt_loc = template.replace("datasetname",dataset_names[i]).replace("datasettype",dataset_types[j])
                txt_paths.append(txt_loc)
            else:
                print("Skipping dataset " + dataset_names[i] + " with " + dataset_types[j])
                continue
    return get_tar_path_from_txts(txt_paths, islocal=islocal)
    
def get_tar_path_from_txts(txt_path, islocal):
    """
    Get tar path from txt path
    """
    if isinstance(txt_path, (list,tuple)):
        print("txt_path",txt_path)
        print([get_tar_path_from_txts(txt_path[i], islocal=islocal) for i in range(len(txt_path))])
        return sum([get_tar_path_from_txts(txt_path[i], islocal=islocal) for i in range(len(txt_path))], [])
    if isinstance(txt_path, str):
        with open(txt_path) as f:
            lines = f.readlines()
        if islocal:
            lines = [lines[i].split("\n")[0].replace("s3://laion-audio/", "/mnt/audio_clip/") for i in range(len(lines))]
        else:
            lines = [lines[i].split("\n")[0] for i in range(len(lines))]
        return lines
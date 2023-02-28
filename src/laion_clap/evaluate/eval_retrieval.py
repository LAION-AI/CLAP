import os.path
import glob
import random
import numpy as np
import logging
import wandb
import torch
import torch.backends.cudnn as cudnn
from laion_clap import create_model
from laion_clap.training.logger import setup_logging
from laion_clap.training.data import get_data
from laion_clap.training.train import evaluate
from laion_clap.utils import get_tar_path_from_dataset_name, dataset_split
from laion_clap.training.params import parse_args


def find_params_value(file, key):
    # find value of params in params_file
    with open(file, 'r') as f:
        for line in f:
            if key + ': ' in line:
                return line.split(': ')[1].strip()
    return None


if __name__ == '__main__':
    # (yusong) repeated run might have different metric results.
    # This is because we randomly select crop 10s for each audio.
    args = parse_args()

    if os.path.isdir(args.pretrained):
        log_dir = os.path.dirname(args.pretrained)
    else:
        log_dir = os.path.dirname(os.path.dirname(args.pretrained))

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_path = os.path.join(log_dir, 'out.log')
    setup_logging(log_path, args.log_level)
    params_file = os.path.join(log_dir, 'params.txt')

    seed = 3407
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = False
    pretrained = 'openai'
    amodel = find_params_value(params_file, 'amodel')
    tmodel = find_params_value(params_file, 'tmodel')

    if amodel is None or tmodel is None:
        raise ValueError('model type not found in params file')

    # set up dummy values for args
    args.parallel_eval = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.val_frequency = 1
    args.epochs = 1
    args.precision = 'fp32'
    args.save_logs = True
    args.wandb = True
    args.class_index_dict = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    if args.remotedata:
        for dataset_name in args.datasetnames:
            for split in dataset_split[dataset_name]:
                if not os.path.exists(f"./json_files/{dataset_name}/{split}"):
                    os.makedirs(f"./json_files/{dataset_name}/{split}")
                os.system(
                    f"aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json ./json_files/{dataset_name}/{split}/sizes.json"
                )

    if args.datasetinfos is None:
        args.datasetinfos = ["train", "unbalanced_train", "balanced_train"]
    if args.dataset_type == "webdataset":
        args.train_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            args.datasetinfos,
            islocal=not args.remotedata,
            proportion=args.dataset_proportion,
            dataset_path=args.datasetpath,
        )
        args.val_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            ["valid", "test", "eval"],
            islocal=not args.remotedata,
            proportion=1,
            dataset_path=args.datasetpath,
        )
    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision='fp32',
        device=device,
        jit=False,
        force_quick_gelu=False,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
        skip_params=False,
        enable_fusion=args.enable_fusion,
        fusion_type=args.fusion_type
    )  # a hack to get model_cfg

    data = get_data(args, model_cfg=model_cfg)  # (yusong): hack: no model_cfg needed to get data

    writer = None  # if use tensorboard, initalize writer here

    if args.wandb:
        assert wandb is not None, "Please install wandb."

        # # find the line with "wandb_notes" and get the value
        # wandb_notes = find_params_value(params_file, 'wandb_notes')
        # if wandb_notes is None:
        #     print(f'wandb_notes not found in params file: {params_file}, set to timestamp.')
        #     wandb_notes = f'experiment_{time.strftime("%Y%m%d-%H%M%S")}'
        # wandb_notes = wandb_notes + '-eval-retrieval'
        wandb_notes = args.wandb_notes

        logging.debug("Starting wandb.")
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        if args.wandb_id is not None:
            wandb.init(
                project="clap",
                id=args.wandb_id,
                resume=True
            )
        else:
            wandb.init(
                project="clap",
                notes=wandb_notes,
                name=wandb_notes,
                tags=[],
                config=vars(args),
            )
        logging.debug("Finished loading wandb.")

    if os.path.isdir(args.pretrained):
        all_model_checkpoints = sorted(glob.glob(os.path.join(log_dir, 'checkpoints', '*.pt')), key=os.path.getmtime)
    else:
        all_model_checkpoints = [args.pretrained]
    for model_path in all_model_checkpoints:
        args.checkpoint_path = os.path.dirname(model_path)
        model, model_cfg = create_model(
            amodel,
            tmodel,
            pretrained,
            precision='fp32',
            device=device,
            jit=False,
            force_quick_gelu=False,
            openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
            skip_params=False,
            enable_fusion=args.enable_fusion,
            fusion_type=args.fusion_type
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
            logging.info(
                f"=> resuming checkpoint '{model_path}' (epoch {start_epoch})"
            )
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            start_epoch = 0

        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        evaluate(model, data, start_epoch, args, writer)

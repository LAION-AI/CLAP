from open_clip import create_model_and_transforms, trace_model, create_model
from training.data import get_data
from training.params import parse_args
import torch
import os
from tqdm import tqdm
from training.distributed import is_master, init_distributed_device, world_info_from_env

args = parse_args()
# sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
args.amodel = args.amodel.replace("/", "-")
device = torch.device('cpu')

# discover initial world args early so we can log properly
args.distributed = False
args.local_rank, args.rank, args.world_size = world_info_from_env()

model, model_cfg = create_model(
    args.amodel,
    args.tmodel,
    args.pretrained,
    precision=args.precision,
    device=device,
    jit=args.torchscript,
    force_quick_gelu=args.force_quick_gelu,
    openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
    skip_params=True,
    pretrained_audio=args.pretrained_audio,
    pretrained_text=args.pretrained_text,
    enable_fusion=args.enable_fusion,
    fusion_type=args.fusion_type
)

data = get_data(args, model_cfg)

dataloader, sampler = data["train"].dataloader, data["train"].sampler

print('dataset size:', data["train"].dataloader.num_samples)
print('batch size:', args.batch_size)
print('num batches:', data["train"].dataloader.num_samples // args.batch_size)

for i, batch in enumerate(tqdm(dataloader, total=data["train"].dataloader.num_samples // args.batch_size)):
    pass

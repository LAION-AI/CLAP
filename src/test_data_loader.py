from open_clip import create_model_and_transforms, trace_model, create_model
from training.data import get_data
from training.params import parse_args
import torch
import os
from tqdm import tqdm

args = parse_args()
# sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
args.amodel = args.amodel.replace("/", "-")
device = torch.device('cpu')

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

for i, batch in enumerate(tqdm(dataloader)):
    pass

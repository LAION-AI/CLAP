from training.data import *
from easydict import EasyDict as edict
from open_clip import create_model_and_transforms, trace_model, create_model
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.params import parse_args

#     --save-frequency 1 \
#     --zeroshot-frequency 1 \
#     --dataset-type="toy" \
#     --train-data="/mnt/audio_clip/test/data/indexes/balanced_train.h5"  \
#     --val-data="/mnt/audio_clip/test/data/indexes/eval.h5"  \
#     --train-ipc="/mnt/audio_clip/test/data/train_ipc.npy" \
#     --val-ipc="/mnt/audio_clip/test/data/val_ipc.npy" \
#     --csv-img-key filepath \
#     --csv-caption-key title \
#     --precision="fp32" \
#     --warmup 10000 \
#     --batch-size=32 \
#     --lr=1e-3 \
#     --wd=0.1 \
#     --epochs=30 \
#     --workers=8 \
#     --model PANN-14
args = parse_args()
device = init_distributed_device(args)
args.train_data = ["/mnt/audio_clip/webdataset_tar/audioset/unbalanced_train/0.tar", "/mnt/audio_clip/webdataset_tar/audioset/unbalanced_train/1.tar"]
model, model_cfg = create_model(
    args.model,
    args.pretrained,
    precision=args.precision,
    device=device,
    jit=args.torchscript,
    force_quick_gelu=args.force_quick_gelu
)
    
data = get_wds_dataset(args, model_cfg, is_train=True)

dl = data.dataloader
for i, batch in enumerate(dl):
    # print(i, batch)
    # print("batch.keys", batch.keys())
    # # 'hdf5_path', 'index_in_hdf5', 'audio_name', 'waveform', 'target', 'text'
    # print("hdf5_path", batch["hdf5_path"])
    # print("index_in_hdf5", batch["index_in_hdf5"])
    # print("audio_name", batch["audio_name"])
    # print("waveform", batch["waveform"])
    # print("waveform.shape", batch["waveform"].shape)
    # print("target", batch["target"])
    # print("target.shape", batch["target"].shape)
    # print("text", batch["text"])
    # print("text.shape", batch["text"].shape)
    print("osr", batch[-1])
    print()
    break 
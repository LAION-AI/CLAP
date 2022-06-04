import json
import logging
import math
import os
import time
from contextlib import suppress
from tkinter import N

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as audioF
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .data import wds_batch_list2dict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # for toy dataset
    if args.dataset_type == 'toy':
        dataloader.dataset.generate_queue()

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        # if args.dataset_type == 'webdataset':
        #     batch = wds_batch_list2dict(batch)
        #     batch["text"] = batch["text"][:,0,:]
        step = num_batches_per_epoch * epoch + i
        if isinstance(scheduler, dict):
            for s in scheduler.values():
                s(step)
        else:
            scheduler(step)
        audios = batch[2]  # (yusong) todo:  change to retrieve from index for now.
        #if args.resample_method=="TorchAudio":
            # kaiser_best
        #    audios = audioF.resample(
        #        audios,
        #        batch["audio_orig_sr"][0],
        #        32000,
        #        lowpass_filter_width=64,
        #        rolloff=0.9475937167399596,
        #        resampling_method="kaiser_window",
        #    )
        texts = batch[3][:,0,:]
        audios = audios.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        if isinstance(optimizer, dict):
            for o_ in optimizer.values():
                o_.zero_grad()
        else:
            optimizer.zero_grad()

        with autocast():
            audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t = model(audios, texts)
            total_loss = loss(audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t)
        if isinstance(optimizer, dict):
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    for o_ in optimizer.values():
                        if args.horovod:
                            o_.synchronize()
                            scaler.unscale_(o_)
                            with o_.skip_synchronize():
                                scaler.step(o_)
                        else:
                            scaler.step(o_)
                    scaler.update()
                else:
                    total_loss.backward()
                    for o_ in optimizer.values():
                        o_.step()
        else:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale_a.clamp_(0, math.log(100))
            unwrap_model(model).logit_scale_t.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(audios)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar_a = logit_scale_a.item()
            logit_scale_scalar_t = logit_scale_t.item()
            if isinstance(optimizer, dict):
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} "
                    f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                    f"Logit Scale Text: {logit_scale_scalar_t:.3f}"
                )
                log_data = {
                    "loss": loss_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "scale_audio":  logit_scale_scalar_a,
                    "scale_text":  logit_scale_scalar_t,
                    "lr": [o_.param_groups[0]["lr"] for o_ in optimizer.values()],
                }
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                    f"Logit Scale Text: {logit_scale_scalar_t:.3f}"
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": loss_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "scale_audio":  logit_scale_scalar_a,
                    "scale_text":  logit_scale_scalar_t,
                    "lr": optimizer.param_groups[0]["lr"]
                }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    # CHANGE
    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    # metrics.update(zero_shot_metrics)

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_audio_features @ all_text_features will blow up memory and compute very quickly
        eval_info = {}
        eval_info["all"] = {"cumulative_loss": 0.0, "num_samples": 0, "all_audio_features": [], "all_text_features": [],
                            "all_audio_features_mlp": [], "all_text_features_mlp": []}        # cumulative_loss = 0.0
        # all_audio_features, all_text_features, all_audio_features_mlp, all_text_features_mlp = [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audios = batch[2]  # (yusong) todo:  change to retrieve from index for now.
                texts = batch[3][:, 0, :]
                audios = audios.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                all_names = list(set(["-".join(b.split("/")[-3:-1]) for b in batch[0]]))
                for name in all_names:
                    if name not in eval_info.keys():
                        eval_info[name] = {"cumulative_loss": 0.0, "num_samples": 0, "all_audio_features": [],
                                           "all_text_features": [], "all_audio_features_mlp": [],
                                           "all_text_features_mlp": []}

                with autocast():
                    audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t = model(audios, texts)

                    num_samples += audio_features.shape[0]

                    for n in [*all_names, "all"]:
                        if n == 'all':
                            idx = list(range(len(batch[0])))
                        else:
                            idx = np.where(np.array(["-".join(b.split("/")[-3:-1]) for b in batch[0]]) == n)[0]
                        eval_info[n]["all_audio_features"].append(
                            audio_features.cpu().index_select(0, torch.tensor(idx).long()))
                        eval_info[n]["all_text_features"].append(
                            text_features.cpu().index_select(0, torch.tensor(idx).long()))
                        eval_info[n]["all_audio_features_mlp"].append(
                            audio_features_mlp.cpu().index_select(0, torch.tensor(idx).long()))
                        eval_info[n]["all_text_features_mlp"].append(
                            text_features_mlp.cpu().index_select(0, torch.tensor(idx).long()))

                # cumulative_loss += total_loss * batch_size
                # num_samples += batch_size
                if is_master(args) and (i % 100) == 0 and i != 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]")
            val_metrics_s = {}
            for n in eval_info.keys():
                metrics_single_dataset = get_metrics(
                    audio_features=torch.cat(eval_info[n]["all_audio_features"]),
                    text_features=torch.cat(eval_info[n]["all_text_features"]),
                    audio_features_mlp=torch.cat(eval_info[n]["all_audio_features_mlp"]),
                    text_features_mlp=torch.cat(eval_info[n]["all_text_features_mlp"]),
                    logit_scale_a=logit_scale_a.cpu(),
                    logit_scale_t=logit_scale_t.cpu(),
                )
                val_metrics_s[n] = {n+'/'+k: v for k, v in metrics_single_dataset.items()}
                metrics.update(val_metrics_s[n])
                if "epoch" not in metrics.keys():
                    metrics.update({"epoch": epoch})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics

# CHANGE here
def get_metrics(audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t):
    metrics = {}
    a_logits_per_audio = (logit_scale_a * audio_features @ text_features_mlp.t()).detach().cpu()
    a_logits_per_text = a_logits_per_audio.t().detach().cpu()
    t_logits_per_audio = (logit_scale_t * audio_features_mlp @ text_features.t()).detach().cpu()
    t_logits_per_text = t_logits_per_audio.t().detach().cpu()

    labels = torch.arange(audio_features.shape[0]).long()

    total_loss = (
                 F.cross_entropy(a_logits_per_audio, labels) +
                 F.cross_entropy(a_logits_per_text, labels) +
                 F.cross_entropy(t_logits_per_audio, labels) +
                 F.cross_entropy(t_logits_per_text, labels)
                 ) / 4

    metrics[f"cumulative_loss"] = total_loss.item()
    metrics[f"num_samples"] = audio_features.shape[0]

    logits = {"audio_to_text": (a_logits_per_audio + t_logits_per_audio) / 2, "text_to_audio": (a_logits_per_text + t_logits_per_text) / 2}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        # map@10
        metrics[f"{name}_R@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))


    return metrics

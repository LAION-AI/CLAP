import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        audio_features,
        text_features,
        audio_features_mlp, 
        text_features_mlp,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_audio_features = hvd.allgather(audio_features)
            all_text_features = hvd.allgather(text_features)
            all_audio_features_= hvd.allgather(audio_features_mlp)
            all_text_features_mlp = hvd.allgather(text_features_mlp)
        else:
            with torch.no_grad():
                all_audio_features = hvd.allgather(audio_features)
                all_text_features = hvd.allgather(text_features)
                all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                all_text_features_mlp = hvd.allgather(text_features_mlp)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_audio_features = list(all_audio_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_audio_features_mlp = list(all_audio_features_mlp.chunk(world_size, dim=0))
                gathered_text_features_mlp = list(all_text_features_mlp.chunk(world_size, dim=0))
                gathered_audio_features[rank] = audio_features
                gathered_text_features[rank] = text_features
                gathered_audio_features_mlp[rank] = audio_features_mlp
                gathered_text_features_mlp[rank] = text_features_mlp
                all_audio_features = torch.cat(gathered_audio_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                all_audio_features_mlp = torch.cat(gathered_audio_features_mlp, dim=0)
                all_text_features_mlp = torch.cat(gathered_text_features_mlp, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_audio_features = torch.cat(torch.distributed.nn.all_gather(audio_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_audio_features_mlp = torch.cat(torch.distributed.nn.all_gather(audio_features_mlp), dim=0)
            all_text_features_mlp = torch.cat(torch.distributed.nn.all_gather(text_features_mlp), dim=0)
        else:
            gathered_audio_features = [torch.zeros_like(audio_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_audio_features_mlp = [torch.zeros_like(audio_features_mlp) for _ in range(world_size)]
            gathered_text_features_mlp = [torch.zeros_like(text_features_mlp) for _ in range(world_size)]
            dist.all_gather(gathered_audio_features, audio_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_audio_features_mlp, audio_features_mlp)
            dist.all_gather(gathered_text_features_mlp, text_features_mlp)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_audio_features[rank] = audio_features
                gathered_text_features[rank] = text_features
                gathered_audio_features_mlp[rank] = audio_features_mlp
                gathered_text_features_mlp[rank] = text_features_mlp
            all_audio_features = torch.cat(gathered_audio_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_audio_features_mlp = torch.cat(gathered_audio_features_mlp, dim=0)
            all_text_features_mlp = torch.cat(gathered_text_features_mlp, dim=0)

    return all_audio_features, all_text_features, all_audio_features_mlp, all_text_features_mlp


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t):
        device = audio_features.device
        if self.world_size > 1:
            all_audio_features, all_text_features, all_audio_features_mlp, all_text_features_mlp = gather_features(
                audio_features, text_features, audio_features_mlp, text_features_mlp,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                a_logits_per_audio = logit_scale_a * audio_features @ all_text_features_mlp.T
                a_logits_per_text = logit_scale_a * text_features_mlp @ all_audio_features.T
                t_logits_per_audio = logit_scale_t * audio_features_mlp @ all_text_features.T
                t_logits_per_text = logit_scale_t * text_features @ all_audio_features_mlp.T
            else:
                a_logits_per_audio = logit_scale_a * all_audio_features @ all_text_features_mlp.T
                a_logits_per_text = a_logits_per_audio.T
                t_logits_per_audio = logit_scale_t * all_audio_features_mlp @ all_text_features.T
                t_logits_per_text = t_logits_per_audio.T
        else:
            a_logits_per_audio = logit_scale_a * audio_features @ text_features_mlp.T
            a_logits_per_text = logit_scale_a * text_features_mlp @ audio_features.T
            t_logits_per_audio = logit_scale_t * audio_features_mlp @ text_features.T
            t_logits_per_text = logit_scale_t * text_features @ audio_features_mlp.T

        # calculated ground-truth and cache if enabled
        num_logits = a_logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]


        total_loss = (
            F.cross_entropy(a_logits_per_audio, labels) +
            F.cross_entropy(a_logits_per_text, labels) + 
            F.cross_entropy(t_logits_per_audio, labels) +
            F.cross_entropy(t_logits_per_text, labels) 
            ) / 4
        return total_loss

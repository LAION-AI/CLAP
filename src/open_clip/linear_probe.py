import numpy as np
import torch.nn.functional as F
from torch import nn
from .model import MLPLayers

class LinearProbe(nn.Module):
    def __init__(self, model, mlp, freeze, in_ch, out_ch, act = None):
        """
        Args:
            model: nn.Module
            mlp: bool, if True, then use the MLP layer as the linear probe module
            freeze: bool, if Ture, then freeze all the CLAP model's layers when training the linear probe
            in_ch: int, the output channel from CLAP model
            out_ch: int, the output channel from linear probe (class_num)
            act: torch.nn.functional, the activation function before the loss function
        """

        self.model = model
        if mlp:
            self.lp_layer = MLPLayers(units = [in_ch, in_ch * 2, out_ch])
        else:
            self.lp_layer = nn.Linear(in_ch, out_ch)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.act = act

    def forward(self, x):
        """
        Args:
            x: waveform, torch.tensor [batch, t_samples]
                
        Returns:
            class_prob: torch.tensor [batch, class_num]

        """
        x = self.lp_layer(x)
        if self.act is not None:
            x = self.act(x)
        return x


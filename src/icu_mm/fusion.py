"""E3 concatenation of the four E2 pooled representations."""
import torch
from torch import nn
from .models import ModalityEncoders, MODALITIES, masked_pool


class ConcatenationFusion(nn.Module):
    def __init__(self, context_dim, frozen=False):
        super().__init__()
        self.encoders = ModalityEncoders(context_dim)
        self.head = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128,1))
        self.frozen = frozen
        if frozen:
            self.encoders.requires_grad_(False)
            self.encoders.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoders.eval()
        return self

    def encode(self, values, valid):
        streams = self.encoders(values)
        return torch.cat([masked_pool(stream,valid[name]) for stream,name in zip(streams,MODALITIES)],dim=1)

    def forward(self, values, valid):
        pooled = values['embedding'] if 'embedding' in values else self.encode(values,valid)
        return self.head(pooled).squeeze(-1)

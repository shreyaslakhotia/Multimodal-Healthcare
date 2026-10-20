"""Fixed-size modality encoders and synchronous cross-attention fusion.

Frozen DenseNet feature extraction is a separate preprocessing stage. The CXR
stream here receives its 49x1024 frozen features, never pixels or report text.
"""
import math
import torch
from torch import nn

MODALITIES = ('context', 'vitals', 'labs', 'cxr')


def sinusoidal(length, width):
    position = torch.arange(length, dtype=torch.float32)[:, None]
    divisor = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.) / width))
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, length, layers=2, learned_position=False):
        super().__init__()
        self.projection = nn.Linear(input_dim, 128)
        if learned_position:
            self.position = nn.Parameter(torch.randn(length, 128) * .02)
        else:
            self.register_buffer('position', sinusoidal(length, 128))
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, values):
        if values.shape[1:] != (len(self.position), self.projection.in_features):
            raise ValueError('Incorrect token shape')
        return self.transformer(self.projection(values) + self.position)


class ModalityEncoders(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(context_dim, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 128))
        self.vitals = SequenceEncoder(16, 48)
        self.labs = SequenceEncoder(22, 48)
        self.cxr = SequenceEncoder(1024, 49, layers=1, learned_position=True)

    def forward(self, inputs):
        return [self.context(inputs['context']).unsqueeze(1),
                self.vitals(inputs['vitals']), self.labs(inputs['labs']), self.cxr(inputs['cxr'])]


class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.ModuleList([nn.MultiheadAttention(128, 4, dropout=.1, batch_first=True) for _ in MODALITIES])
        self.attention_norm = nn.ModuleList([nn.LayerNorm(128) for _ in MODALITIES])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(128) for _ in MODALITIES])
        self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 128), nn.Dropout(.1)) for _ in MODALITIES])
        self.dropout = nn.Dropout(.1)

    def forward(self, streams, present):
        updated = []
        for modality in range(4):
            others = [i for i in range(4) if i != modality]
            keys = torch.cat([streams[i] for i in others], dim=1)
            key_present = torch.cat([present[:, i:i+1].expand(-1, streams[i].shape[1]) for i in others], dim=1)
            # If every other modality is missing, expose a zero null token.
            # Otherwise it stays masked. This avoids all-masked softmax NaNs.
            null = torch.zeros_like(keys[:, :1])
            keys = torch.cat([keys, null], dim=1)
            null_present = ~key_present.any(dim=1, keepdim=True)
            key_present = torch.cat([key_present, null_present], dim=1)
            # All updates read the same previous streams, not in-place updates.
            result, _ = self.attention[modality](streams[modality], keys, keys,
                                               key_padding_mask=~key_present,
                                               need_weights=False)
            result = self.attention_norm[modality](streams[modality] + self.dropout(result))
            updated.append(self.ffn_norm[modality](result + self.ffns[modality](result)))
        return updated


class CrossAttentionFusion(nn.Module):
    def __init__(self, context_dim, modality_dropout=0.):
        super().__init__()
        if not 0 <= modality_dropout <= 1:
            raise ValueError('Modality dropout must be in [0,1]')
        self.encoders = ModalityEncoders(context_dim)
        self.modality_type = nn.Parameter(torch.randn(4, 128) * .02)
        self.missing = nn.Parameter(torch.randn(4, 128) * .02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(), CrossAttentionBlock()])
        self.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 1))
        self.modality_dropout = modality_dropout

    def forward(self, inputs, present=None):
        batch = inputs['context'].shape[0]
        device = inputs['context'].device
        if present is None:
            present = torch.ones(batch, 4, dtype=torch.bool, device=device)
        else:
            present = present.to(device=device, dtype=torch.bool).clone()
        if present.shape != (batch, 4):
            raise ValueError('Expected presence mask [batch,4]')
        if self.training and self.modality_dropout:
            keep = torch.rand(batch, 3, device=device) >= self.modality_dropout
            present[:, 1:] &= keep
        # Remove hidden modality inputs before encoding as well as masking keys.
        # This prevents missing values/NaNs from affecting any visible stream.
        safe_inputs = {}
        for i, name in enumerate(MODALITIES):
            dims = [batch] + [1] * (inputs[name].ndim - 1)
            safe_inputs[name] = torch.where(present[:, i].reshape(dims), inputs[name], 0.)
        streams = self.encoders(safe_inputs)
        streams = [stream + self.modality_type[i] for i, stream in enumerate(streams)]
        for block in self.blocks:
            streams = block(streams, present)
        pooled = [torch.where(present[:, i:i+1], stream.mean(dim=1), self.missing[i])
                  for i, stream in enumerate(streams)]
        return self.head(torch.cat(pooled, dim=1)).squeeze(-1)

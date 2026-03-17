
"""
Local prototype detector with energy priors for EELS.

Encoder produces hidden features; per-element learnable prototypes match local patterns;
energy priors gate where each element looks; evidence is aggregated via masked pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    x:    [B, C, N]
    mask: [B, 1, N] or [B, C, N]
    ->    [B, C]
    """
    if mask.dim() == 3 and mask.size(1) == 1:
        mask = mask.expand_as(x)
    mask = mask.to(x.dtype)
    s = (x * mask).sum(dim=dim)
    d = mask.sum(dim=dim).clamp_min(eps)
    return s / d


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    x:    [B, C, N]
    mask: [B, 1, N] or [B, C, N]
    ->    [B, C]
    """
    if mask.dim() == 3 and mask.size(1) == 1:
        mask = mask.expand_as(x)
    very_neg = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(mask == 0, very_neg)
    out = x_masked.max(dim=dim).values
    valid = (mask.sum(dim=dim) > 0)
    out = torch.where(valid, out, torch.zeros_like(out))
    return out


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    scores: [B, C, N]
    mask:   [B, 1, N] or [B, C, N]
    """
    if mask.dim() == 3 and mask.size(1) == 1:
        mask = mask.expand_as(scores)
    mask = mask.to(scores.dtype)

    very_neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(mask == 0, very_neg)

    m = scores.max(dim=dim, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))

    e = torch.exp(scores - m) * mask
    s = e.sum(dim=dim, keepdim=True)
    out = e / (s + eps)
    out = torch.where(s > 0, out, torch.zeros_like(out))
    return out


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, p: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, k: int = 5, p: float = 0.0):
        super().__init__()
        self.block1 = ConvBlock(channels, channels, k, p=p)
        self.block2 = ConvBlock(channels, channels, k, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


class EELSModel(nn.Module):
    """
    Multi-label EELS model.

    Input:
        x    : [B, 1, N]
        mask : [B, 1, N]

    Output dict:
        logits      : [B, num_classes]
        probs       : [B, num_classes]
        attention   : [B, num_classes, N]
        class_map   : [B, num_classes, N]
        pooled_att  : [B, num_classes]
        pooled_max  : [B, num_classes]
        pooled_mean : [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 80,
        spectrum_length: int = 3072,
        hidden: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.spectrum_length = spectrum_length
        self.hidden = hidden

        # Input channels: x, mask, coord, dx
        self.stem1 = ConvBlock(4, 64, 7, p=dropout)
        self.stem2 = ConvBlock(64, hidden, 7, p=dropout)
        self.stem3 = ConvBlock(hidden, hidden, 5, p=dropout)

        self.res1 = ResidualBlock(hidden, k=5, p=dropout)
        self.res2 = ResidualBlock(hidden, k=5, p=dropout)

        # Multi-scale branch
        self.ms3 = ConvBlock(hidden, hidden, 3, p=dropout)
        self.ms7 = ConvBlock(hidden, hidden, 7, p=dropout)
        self.ms15 = ConvBlock(hidden, hidden, 15, p=dropout)
        self.merge = ConvBlock(hidden * 3, hidden, 1, p=dropout)

        # Per-class local evidence
        self.class_map_head = nn.Conv1d(hidden, num_classes, kernel_size=1)

        # Per-class attention scores
        self.attn_head = nn.Conv1d(hidden, num_classes, kernel_size=1)

        # Class-wise linear combiner for [att, max, mean]
        self.logit_scale = nn.Parameter(torch.ones(num_classes, 3))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EELSModel initialized with {num_params} trainable parameters.")

    def init_bias_from_prevalence(self, prevalence: torch.Tensor) -> None:
        """Set logit_bias[k] = log(p_k / (1 - p_k)) with p_k clipped to [1e-4, 1-1e-4]."""
        p = prevalence.to(self.logit_bias.dtype).clamp(1e-4, 1.0 - 1e-4)
        self.logit_bias.data.copy_(torch.log(p / (1.0 - p)))

    def _build_input_channels(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, 1, N]
        mask: [B, 1, N]
        returns: [B, 4, N]
        """
        b, _, n = x.shape
        device = x.device
        dtype = x.dtype

        coord = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)
        coord = coord.view(1, 1, n).expand(b, 1, n)

        dx = torch.zeros_like(x)
        dx[..., 1:] = x[..., 1:] - x[..., :-1]

        inp = torch.cat([x, mask.to(dtype), coord, dx], dim=1)
        return inp

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        x:    [B, 1, N]
        mask: [B, 1, N]
        """
        inp = self._build_input_channels(x, mask)

        # Backbone
        h = self.stem1(inp)
        h = self.stem2(h)
        h = self.stem3(h)

        h = self.res1(h)
        h = self.res2(h)

        h3 = self.ms3(h)
        h7 = self.ms7(h)
        h15 = self.ms15(h)
        h = torch.cat([h3, h7, h15], dim=1)
        h = self.merge(h)  # [B, hidden, N]

        # Per-class evidence over energy
        class_map = self.class_map_head(h)   # [B, C, N]

        # Per-class attention over energy
        attn_scores = self.attn_head(h)      # [B, C, N]
        attention = masked_softmax(attn_scores, mask, dim=-1)  # [B, C, N]

        if mask.dim() == 3 and mask.size(1) == 1:
            class_mask = mask.expand_as(class_map)
        else:
            class_mask = mask

        # Three pooled summaries per class
        pooled_att = (class_map * attention).sum(dim=-1)         # [B, C]
        pooled_max = masked_max(class_map, class_mask, dim=-1)   # [B, C]
        pooled_mean = masked_mean(class_map, class_mask, dim=-1) # [B, C]

        pooled = torch.stack([pooled_att, pooled_max, pooled_mean], dim=-1)  # [B, C, 3]

        # Independent class logits
        logits = (pooled * self.logit_scale.unsqueeze(0)).sum(dim=-1) + self.logit_bias.unsqueeze(0)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probs": probs,
            "attention": attention,
            "class_map": class_map,
            "pooled_att": pooled_att,
            "pooled_max": pooled_max,
            "pooled_mean": pooled_mean,
        }
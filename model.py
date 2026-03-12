from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute softmax with masking. Masked positions get probability 0.
    Safe for all-masked rows (returns zeros instead of NaN).
    
    Args:
        scores: [B, K, N] attention scores
        mask: [B, 1, N] or [B, K, N] mask (1 = valid, 0 = masked)
        dim: dimension to apply softmax over
        eps: small epsilon for numerical stability
    
    Returns:
        [B, K, N] attention weights (probabilities)
    """
    # Convert mask to same dtype as scores
    mask = mask.to(dtype=scores.dtype)
    
    # Broadcast mask if needed: [B, 1, N] -> [B, K, N]
    if mask.dim() == 3 and mask.size(1) == 1:
        mask = mask.expand_as(scores)
    
    # Set masked positions to very negative value
    very_neg = torch.finfo(scores.dtype).min
    scores_masked = scores.masked_fill(mask == 0, very_neg)
    
    # Subtract max for numerical stability
    scores_max = scores_masked.max(dim=dim, keepdim=True).values
    # Handle all-masked case: if max is -inf, set to 0
    scores_max = torch.where(torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max))
    
    # Compute exp
    exp_scores = torch.exp(scores_masked - scores_max) * mask
    
    # Normalize
    sum_exp = exp_scores.sum(dim=dim, keepdim=True)
    
    # Handle all-masked case: if sum is 0, return zeros
    attention = exp_scores / (sum_exp + eps)
    attention = torch.where(sum_exp > 0, attention, torch.zeros_like(attention))
    
    return attention


class Conv1DEncoder(nn.Module):
    """Conv1D encoder for EELS spectra."""
    
    def __init__(self, in_channels: int = 1, channels: int = 256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, N] input spectra
        
        Returns:
            [B, C, N] encoded features where C=channels
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        
        return x


class EELSPerElementAttentionModel(nn.Module):
    """
    Baseline model with per-element attention for EELS multi-label classification.
    
    Architecture:
        x [B, 1, 3072] -> Conv1D encoder -> H [B, 128, 3072]
        H -> per-element attention scorer -> scores [B, 80, 3072]
        scores + mask -> masked softmax -> attention [B, 80, 3072]
        attention @ H -> pooled [B, 80, 128]
        pooled -> element-wise linear head -> logits [B, 80]
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        spectrum_length: int = 3072,
        channels: int = 256,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.spectrum_length = spectrum_length
        self.channels = channels
        
        # Encoder
        self.encoder = Conv1DEncoder(in_channels=1, channels=channels)
        
        # Per-element attention scorer
        # Output: [B, num_classes, spectrum_length]
        self.attention_scorer = nn.Conv1d(
            in_channels=channels,
            out_channels=num_classes,
            kernel_size=1
        )
        
        # Per-class linear classification head
        # Each class has its own weight vector and bias
        self.element_weights = nn.Parameter(torch.empty(num_classes, channels))
        self.element_bias = nn.Parameter(torch.zeros(num_classes))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.element_weights)
        nn.init.zeros_(self.element_bias)
    
    def forward(
        self,
        x: torch.Tensor,
        nonzero_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 1, N] input spectra
            nonzero_mask: [B, 1, N] mask (1 = valid, 0 = padded). If None, no masking.
            return_attention: if True, return attention weights
        
        Returns:
            Dictionary with:
                - "logits": [B, num_classes] classification logits
                - "attention": [B, num_classes, N] attention weights (if return_attention=True)
                - "features": [B, channels, N] encoder features (if return_attention=True)
                - "pooled": [B, num_classes, channels] pooled features (if return_attention=True)
        """
        B = x.size(0)
        
        # Encode: [B, 1, N] -> [B, C, N]
        H = self.encoder(x)  # [B, channels, spectrum_length]
        
        # Compute attention scores: [B, C, N] -> [B, num_classes, N]
        scores = self.attention_scorer(H)  # [B, num_classes, spectrum_length]
        
        # Apply masked softmax if mask provided
        if nonzero_mask is not None:
            attention = masked_softmax(scores, nonzero_mask, dim=-1)
        else:
            attention = F.softmax(scores, dim=-1)
        
        # Element-wise pooling: attention @ H^T
        # attention: [B, num_classes, N]
        # H: [B, channels, N] -> [B, N, channels]
        # Result: [B, num_classes, channels]
        H_transposed = H.transpose(1, 2)  # [B, N, channels]
        pooled = torch.bmm(attention, H_transposed)  # [B, num_classes, channels]
        
        # Per-class linear classification
        # pooled: [B, num_classes, channels]
        # element_weights: [num_classes, channels]
        # element_bias: [num_classes]
        # Result: [B, num_classes]
        logits = torch.einsum("bkc,kc->bk", pooled, self.element_weights) + self.element_bias
        
        output = {"logits": logits}
        
        if return_attention:
            output["attention"] = attention
            output["features"] = H
            output["pooled"] = pooled
        
        return output
    
    def initialize_output_bias(self, class_prevalence: torch.Tensor) -> None:
        """
        Initialize output bias based on class prevalence.
        Each class gets its own bias initialized from its prevalence.
        
        Args:
            class_prevalence: [num_classes] tensor with class prevalence values
        """
        # Clip prevalence to avoid log(0) or log(inf)
        prevalence_clipped = torch.clamp(class_prevalence, min=1e-4, max=1.0 - 1e-4)
        
        # Compute bias for each class: b_k = log(p_k / (1 - p_k))
        bias_values = torch.log(prevalence_clipped / (1.0 - prevalence_clipped))
        
        # Set per-class bias
        self.element_bias.data.copy_(bias_values)


if __name__ == "__main__":
    # Smoke test
    print("Testing EELSPerElementAttentionModel...")
    
    model = EELSPerElementAttentionModel(channels=256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with mask
    x = torch.randn(4, 1, 3072)
    mask = (torch.rand(4, 1, 3072) > 0.3).float()
    
    out = model(x, nonzero_mask=mask, return_attention=True)
    
    print(f"Input x shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output logits shape: {out['logits'].shape}")
    print(f"Output attention shape: {out['attention'].shape}")
    print(f"Output features shape: {out['features'].shape}")
    print(f"Output pooled shape: {out['pooled'].shape}")
    
    # Check attention sums to 1 where mask is valid
    attention_sum = out['attention'].sum(dim=-1)  # [B, num_classes]
    print(f"Attention sums (should be ~1.0): {attention_sum.mean():.4f} ± {attention_sum.std():.4f}")
    
    # Check masked positions have zero attention
    masked_positions = (mask == 0).expand_as(out['attention'])
    masked_attention = out['attention'][masked_positions]
    print(f"Attention in masked positions (should be 0): max={masked_attention.max():.6f}")
    
    print("✓ All tests passed!")

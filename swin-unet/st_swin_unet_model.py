"""
Spatio-Temporal Swin-UNet (st-Swin-UNet) for River Morphology Prediction.

This model adapts Swin Transformer to a U-Net architecture for binary segmentation
of satellite imagery, predicting water/non-water pixels from temporal sequences.

Uses torchvision's Swin Transformer components (requires torchvision >= 0.12.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List

try:
    from torchvision.models.swin_transformer import (
        SwinTransformerBlock,
        PatchMerging,
    )
except ImportError:
    raise ImportError("Please install torchvision >= 0.12.0 to use SwinTransformer components.")


class SpatioTemporalPatchEmbed(nn.Module):
    """
    Spatio-Temporal Patch Embedding with learnable temporal position embeddings.

    Each temporal frame is embedded separately using a shared projection,
    temporal position embeddings are added, then frames are aggregated.

    This is the standard approach used in video transformers (ViViT, TimeSformer).

    Input:  (B, T, H, W) - T frames of single-channel images
    Output: (B, H/patch_size, W/patch_size, embed_dim)
    """
    def __init__(
        self,
        patch_size: List[int] = [4, 4],
        num_frames: int = 4,
        embed_dim: int = 96,
        norm_layer: Optional[Callable] = None,
        temporal_aggregation: str = "concat_proj",  # "concat_proj", "learned_weighted_sum", or "mean"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.temporal_aggregation = temporal_aggregation

        # Shared projection for each frame (single channel per frame)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable temporal position embeddings - one per frame
        # Shape: (1, T, 1, 1, embed_dim) - broadcasts across batch and spatial dims
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Temporal aggregation method
        if temporal_aggregation == "concat_proj":
            # Concatenate all frames and project back to embed_dim
            self.temporal_proj = nn.Sequential(
                nn.Linear(num_frames * embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        elif temporal_aggregation == "learned_weighted_sum":
            # Learn weights for each frame (like attention but simpler)
            self.temporal_weights = nn.Parameter(torch.ones(num_frames) / num_frames)
        # "mean" requires no extra parameters

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W) input tensor with T temporal frames
        Returns:
            (B, H/P, W/P, embed_dim) embedded patches
        """
        B, T, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"

        # Embed each frame separately using shared projection
        # (B, T, H, W) -> (B*T, 1, H, W)
        x = x.reshape(B * T, 1, H, W)
        x = self.proj(x)  # (B*T, embed_dim, H/P, W/P)

        _, _, Hp, Wp = x.shape

        # Reshape to (B, T, H/P, W/P, embed_dim)
        x = x.view(B, T, self.embed_dim, Hp, Wp)
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, Hp, Wp, embed_dim)

        # Add temporal position embeddings
        x = x + self.temporal_pos_embed  # Broadcasting: (B, T, Hp, Wp, embed_dim)

        # Aggregate temporal dimension
        if self.temporal_aggregation == "concat_proj":
            # (B, T, Hp, Wp, embed_dim) -> (B, Hp, Wp, T*embed_dim) -> (B, Hp, Wp, embed_dim)
            x = x.permute(0, 2, 3, 1, 4)  # (B, Hp, Wp, T, embed_dim)
            x = x.reshape(B, Hp, Wp, T * self.embed_dim)
            x = self.temporal_proj(x)
        elif self.temporal_aggregation == "learned_weighted_sum":
            # Softmax over temporal weights for stability
            weights = F.softmax(self.temporal_weights, dim=0)  # (T,)
            # Weighted sum: (B, T, Hp, Wp, embed_dim) * (T,) -> (B, Hp, Wp, embed_dim)
            x = (x * weights.view(1, T, 1, 1, 1)).sum(dim=1)
        else:  # "mean"
            x = x.mean(dim=1)  # (B, Hp, Wp, embed_dim)

        x = self.norm(x)
        return x


class PatchExpanding(nn.Module):
    """
    Patch expanding layer (inverse of PatchMerging).

    Upsamples spatial resolution by 2x and halves channel dimension.
    This is the exact inverse of torchvision's PatchMerging.

    Input:  (B, H, W, C)
    Output: (B, 2*H, 2*W, C/2)
    """
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # Expand channels for 2x2 spatial upsampling: need 2*2*(dim//2) = 2*dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        x = self.expand(x)  # (B, H, W, 2*C)

        # Reshape for pixel shuffle: 2*C = 2 * 2 * (C/2)
        x = x.view(B, H, W, 2, 2, C // 2)

        # Interleave to double spatial dimensions
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)

        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Contains multiple SwinTransformerBlocks with alternating window shifts.
    """
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: List[int],
                 mlp_ratio: float = 4.0, dropout: float = 0.0, attention_dropout: float = 0.0,
                 stochastic_depth_prob: float = 0.0, norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth

        if isinstance(stochastic_depth_prob, (list, tuple)):
            sd_probs = stochastic_depth_prob
        else:
            sd_probs = [stochastic_depth_prob] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_depth_prob=sd_probs[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class StSwinUnet(nn.Module):
    """
    Spatio-Temporal Swin-UNet for river morphology prediction.

    Architecture:
    - Spatio-temporal patch embedding with learnable temporal position embeddings
    - Swin Transformer encoder with hierarchical feature extraction
    - Symmetric decoder with skip connections
    - Final head for binary segmentation

    Args:
        img_size: Expected input image size (H, W). Documentation only.
        patch_size: Size of each patch for patch embedding.
        in_chans: Number of input channels (temporal frames, e.g., 4 years).
        num_classes: Number of output classes (1 for binary segmentation).
        embed_dim: Base embedding dimension.
        depths: Number of Swin blocks at each stage.
        num_heads: Number of attention heads at each stage.
        window_size: Window size for local attention.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        temporal_aggregation: How to aggregate temporal frames after adding position embeddings.
            - "concat_proj": Concatenate and project (most expressive)
            - "learned_weighted_sum": Learnable weighted sum (lightweight)
            - "mean": Simple average (baseline)
    """
    def __init__(
        self,
        img_size: tuple = (1000, 500),
        patch_size: List[int] = [4, 4],
        in_chans: int = 4,
        num_classes: int = 1,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: List[int] = [7, 7],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Callable = nn.LayerNorm,
        temporal_aggregation: str = "concat_proj",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Dimensions at each stage: embed_dim * 2^i
        self.stage_dims = [int(embed_dim * 2 ** i) for i in range(self.num_stages)]

        # Stochastic depth decay
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # --- Spatio-Temporal Patch Embedding ---
        self.patch_embed = SpatioTemporalPatchEmbed(
            patch_size=patch_size,
            num_frames=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            temporal_aggregation=temporal_aggregation,
        )

        # --- Encoder ---
        self.encoder_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_stage in range(self.num_stages):
            dim = self.stage_dims[i_stage]
            stage_start = sum(depths[:i_stage])
            stage_end = stage_start + depths[i_stage]
            stage_dpr = dpr[stage_start:stage_end]

            layer = BasicLayer(
                dim=dim,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=drop_rate,
                attention_dropout=attn_drop_rate,
                stochastic_depth_prob=stage_dpr,
                norm_layer=norm_layer,
            )
            self.encoder_layers.append(layer)

            if i_stage < self.num_stages - 1:
                self.downsamples.append(PatchMerging(dim, norm_layer))

        # --- Decoder ---
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()

        for i_dec in range(self.num_stages - 1):
            enc_stage = self.num_stages - 2 - i_dec

            if i_dec == 0:
                in_dim = self.stage_dims[-1]
            else:
                in_dim = self.stage_dims[enc_stage + 1]

            out_dim = self.stage_dims[enc_stage]

            self.upsamples.append(PatchExpanding(in_dim, norm_layer))
            self.skip_fusions.append(nn.Linear(2 * out_dim, out_dim, bias=False))

            layer = BasicLayer(
                dim=out_dim,
                depth=depths[enc_stage],
                num_heads=num_heads[enc_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=drop_rate,
                attention_dropout=attn_drop_rate,
                stochastic_depth_prob=0.0,
                norm_layer=norm_layer,
            )
            self.decoder_layers.append(layer)

        # --- Final Head ---
        self.final_norm = norm_layer(embed_dim)
        self.final_expand = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W) where T is number of temporal frames (years)
        Returns:
            (B, num_classes, H, W) binary segmentation prediction
        """
        B, T, H, W = x.shape

        # Pad to be divisible by patch_size * 2^(num_stages-1)
        factor = self.patch_size[0] * (2 ** (self.num_stages - 1))
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Patch embedding with temporal position embeddings
        x = self.patch_embed(x)  # (B, H/P, W/P, embed_dim)

        # Encoder
        encoder_features = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            encoder_features.append(x)
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        # Decoder with skip connections
        x = encoder_features[-1]
        for i, (upsample, fusion, decoder_layer) in enumerate(
            zip(self.upsamples, self.skip_fusions, self.decoder_layers)
        ):
            x = upsample(x)
            skip = encoder_features[-(i + 2)]
            x = torch.cat([x, skip], dim=-1)
            x = fusion(x)
            x = decoder_layer(x)

        # Final head
        x = self.final_norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, H/P, W/P)
        x = self.final_expand(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        x = torch.sigmoid(x)
        return x


def create_swin_unet_tiny(in_chans: int = 4, num_classes: int = 1,
                          temporal_aggregation: str = "concat_proj", **kwargs) -> StSwinUnet:
    """Create a tiny Swin-UNet suitable for this dataset."""
    return StSwinUnet(
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        temporal_aggregation=temporal_aggregation,
        **kwargs,
    )


def create_swin_unet_small(in_chans: int = 4, num_classes: int = 1,
                           temporal_aggregation: str = "concat_proj", **kwargs) -> StSwinUnet:
    """Create a small Swin-UNet - standard Swin-T configuration."""
    return StSwinUnet(
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        temporal_aggregation=temporal_aggregation,
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing StSwinUnet with Temporal Position Embeddings")
    print("=" * 60)

    # Test all temporal aggregation methods
    for agg_method in ["concat_proj", "learned_weighted_sum", "mean"]:
        print(f"\n--- Temporal Aggregation: {agg_method} ---")

        model = create_swin_unet_tiny(
            in_chans=4,
            num_classes=1,
            temporal_aggregation=agg_method,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Quick test
        x = torch.randn(2, 4, 256, 128)
        with torch.no_grad():
            output = model(x)
        print(f"Input: {x.shape} -> Output: {output.shape}")

        # Check temporal embeddings
        temp_embed = model.patch_embed.temporal_pos_embed
        print(f"Temporal position embedding shape: {temp_embed.shape}")

    # Full size test with recommended method
    print("\n" + "=" * 60)
    print("Full size test (1000x500) with concat_proj")
    print("=" * 60)

    model = create_swin_unet_tiny(in_chans=4, temporal_aggregation="concat_proj")
    x_full = torch.randn(1, 4, 1000, 500)

    with torch.no_grad():
        output = model(x_full)

    print(f"Input: {x_full.shape}")
    print(f"Output: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == (1, 1, 1000, 500), "Shape mismatch!"
    print("\nAll tests passed!")

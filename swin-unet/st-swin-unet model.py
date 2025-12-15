import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable, List

# Importing specific components from torchvision as requested.
# Note: Ensure torchvision >= 0.12.0 is installed.
try:
    from torchvision.models.swin_transformer import (
        SwinTransformerBlock,
        PatchMerging
    )
except ImportError:
    raise ImportError("Please install torchvision >= 0.12.0 to use SwinTransformer components.")

class PatchEmbed(nn.Module):
    """
    Splits the image into patches and embeds them.
    Adapted for 4-channel input (Spatio-Temporal: 4 years).
    """
    def __init__(self, patch_size=[4, 4], in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        # Check constraints
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            # Padding handled externally or via dynamic padding in proj if needed,
            # but usually Swin expects inputs divisible by patch_size.
            # Here we let Conv2d handle it or assume padding is done before.
            pass
            
        x = self.proj(x)  # B, Embed, H/P, W/P
        x = x.permute(0, 2, 3, 1)  # B, H/P, W/P, Embed
        if self.norm:
            x = self.norm(x)
        return x

class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer for the Decoder (Upsampling).
    Inverse of PatchMerging.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # Linear projection to double the feature dimension before reshaping
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        
        x = self.expand(x) # B, H, W, 2*C
        
        # Rearrange for upsampling (Pixel Shuffle logic)
        # We want to go from (H, W, 2C) -> (2H, 2W, C/2)
        # Actually standard expansion in Swin-Unet often does: 
        # Linear(dim -> 4*dim) -> Rearrange -> dim.
        # Let's adjust to match standard Swin-Unet implementation logic.
        # Standard: dim -> 2*dim via Linear, then shuffle.
        # But we need to double resolution. 
        # (H, W, C) -> (H, W, 2*C) -> (2H, 2W, C/2).
        
        x = x.view(B, H, W, 2, 2, C // 4) 
        # Wait, C//4? If input dim is 2*C (after expand), we need output C/2.
        # 2*C = 4 * (C/2). So yes.
        
        x = x.permute(0, 1, 3, 2, 4, 5) # B, H, 2, W, 2, C/4
        x = x.reshape(B, H * 2, W * 2, C // 2) # B, 2H, 2W, C/2
        
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    """
    A basic Swin Layer for one stage (Encoder or Decoder).
    Consists of multiple SwinTransformerBlocks.
    """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., dropout=0., 
                 attention_dropout=0., stochastic_depth_prob=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_depth_prob=stochastic_depth_prob[i] if isinstance(stochastic_depth_prob, list) else stochastic_depth_prob,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class StSwinUnet(nn.Module):
    """
    Spatio-Temporal Swin-Unet (st-Swin-Unet).
    
    Replaces the CNN architecture from the JamUNet thesis.
    Adapts SwinTransformer to a U-Net architecture with 4-channel input (4 years).
    """
    def __init__(self, 
                 img_size=(1000, 500), # Approximate size from thesis
                 patch_size=[4, 4], 
                 in_chans=4, # 4 input images (t-3, t-2, t-1, t)
                 num_classes=1, # Binary segmentation (Water/Non-water)
                 embed_dim=96, 
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7], 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Stochastic depth rules
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # --- Encoder ---
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        
        self.encoder_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            # Swin Blocks for this stage
            dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer(
                dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=drop_rate,
                attention_dropout=attn_drop_rate,
                stochastic_depth_prob=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer
            )
            self.encoder_layers.append(layer)
            
            # Patch Merging (Downsample) - Not added for the last layer of encoder (the bottleneck)
            if i_layer < self.num_layers - 1:
                # PatchMerging from torchvision expects 4*dim -> 2*dim logic
                # We need to instantiate it with input dim. 
                # Note: torchvision's PatchMerging implementation takes `dim` as input dim.
                self.downsamples.append(PatchMerging(dim, norm_layer))

        # --- Decoder ---
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.linear_fusions = nn.ModuleList() # To fuse skip connection + upsampled features

        # Iterate backwards for decoder construction (excluding the bottleneck layer index)
        # Bottleneck output is x_{num_layers-1}
        # Decoder reconstructs from num_layers-2 down to 0
        
        for i_layer in range(self.num_layers - 2, -1, -1):
            input_dim = int(embed_dim * 2 ** (i_layer + 1))
            output_dim = int(embed_dim * 2 ** i_layer)
            
            # Upsampling layer
            self.upsamples.append(PatchExpanding(dim=input_dim, norm_layer=norm_layer))
            
            # Fusion layer (Linear projection) to reduce channel dimension after concatenation
            # Input: Output_dim (from skip) + Output_dim (from upsample) = 2 * Output_dim
            # Output: Output_dim
            self.linear_fusions.append(nn.Linear(2 * output_dim, output_dim, bias=False))
            
            # Swin Blocks for this stage
            layer = BasicLayer(
                dim=output_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=drop_rate,
                attention_dropout=attn_drop_rate,
                stochastic_depth_prob=0., # Typically less stochastic depth in decoder
                norm_layer=norm_layer
            )
            self.decoder_layers.append(layer)

        # --- Final Head ---
        # Final patch expansion to restore original image resolution (4x upsampling from patch embedding)
        self.final_expand = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: (B, 4, H, W)
        
        # 1. Pad input to be multiple of window_size * 2^num_stages
        # Standard Swin requires dimensions divisible by 32 (if window=7, slightly different math, 
        # but torchvision handles padding inside shifted_window_attention).
        # However, patch merging requires divisibility by 2 at each stage.
        H, W = x.shape[2], x.shape[3]
        pad_factor = 2 ** (self.num_layers + 1) # Factor 32 for standard depths
        pad_h = (pad_factor - H % pad_factor) % pad_factor
        pad_w = (pad_factor - W % pad_factor) % pad_factor
        x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 2. Patch Embedding
        x = self.patch_embed(x) # B, H/4, W/4, Embed_dim
        
        # 3. Encoder
        encoder_features = [] # To store skip connections
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            # Store feature for skip connection
            # Note: We need to normalize before storing for skip connection usually? 
            # Or just store the raw output. Swin output is usually pre-norm in V1 block logic?
            # Torchvision implementation: x = x + stochastic_depth(attn(norm1(x)))
            # So x is NOT normalized at the end.
            encoder_features.append(x)
            
            # Downsample (except for the last layer)
            if i < self.num_layers - 1:
                x = self.downsamples[i](x)

        # At this point, x is the bottleneck feature. 
        # encoder_features contains [Stage1_out, Stage2_out, Stage3_out, Stage4_out(Bottleneck)]
        
        # 4. Decoder
        # Iterate backwards. 
        # encoder_features index: 0, 1, 2, 3. 
        # Bottleneck is index 3. 
        # First decoder stage takes Bottleneck, upsamples, fuses with index 2.
        
        bottleneck = encoder_features[-1]
        x = bottleneck
        
        for i, layer in enumerate(self.decoder_layers):
            # Upsample
            x = self.upsamples[i](x)
            
            # Skip connection
            skip = encoder_features[-(i + 2)] # -2, -3, -4 corresponding to indices 2, 1, 0
            
            # Concatenate
            x = torch.cat([x, skip], dim=-1)
            
            # Fuse (Linear projection)
            x = self.linear_fusions[i](x)
            
            # Decoder Swin Blocks
            x = layer(x)

        # 5. Final Head (Restore resolution)
        # x is currently (B, H/4, W/4, embed_dim). Need to go to (B, Num_Classes, H, W)
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        x = self.final_expand(x)
        
        # 6. Unpad to original size
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            
        # 7. Activation (Sigmoid for binary segmentation as per thesis)
        x = torch.sigmoid(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Create model instance
    # Dimensions based on the thesis (approx 1000x500 input)
    model = StSwinUnet(
        img_size=(1000, 500),
        in_chans=4,
        num_classes=1,
        embed_dim=96
    )
    
    # Dummy input: Batch size 1, 4 channels (years), 1000 height, 500 width
    x = torch.randn(1, 4, 1000, 500)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
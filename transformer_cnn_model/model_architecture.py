# This module stores the code for the CNN U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Standard UNet-style block:
    Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU (optionally + Dropout)
    """
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, drop_channels=True, p_drop=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        padding = kernel_size // 2

        layers = [
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if drop_channels and p_drop is not None:
            layers.append(nn.Dropout2d(p=p_drop))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pooling='max', drop_channels=False, p_drop=None):
        super().__init__()
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(2)
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
        self.pool_conv = nn.Sequential(
            self.pooling,
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=True, drop_channels=False, p_drop=None):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class TemporalTransformerBlock(nn.Module):
    """
    Temporal transformer applied along the 'time' dimension (n_channels)
    for each spatial location independently.

    Input:  x of shape (B, T, H, W)
    Output: x of shape (B, T, H, W)   (same shape)
    """
    def __init__(self, T, d_model=16, nhead=4, num_layers=2,
                 dim_feedforward=64, dropout=0.1):
        super().__init__()

        self.T = T
        self.d_model = d_model

        # Project scalar per time-step to d_model
        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # shape (batch, seq, features)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project back to scalar per time-step
        self.output_proj = nn.Linear(d_model, 1)

        # max "batch" size (B*H*W) we feed to the transformer at once
        self.max_tokens = 60000

    def forward(self, x):
        """
        x: (B, T, H, W)
        """
        B, T, H, W = x.shape
        assert T == self.T, f"Expected T={self.T}, but got T={T}"

        # (B, T, H, W) -> (B, H, W, T)
        x = x.permute(0, 2, 3, 1)          # (B, H, W, T)

        # Flatten spatial dims: each (i,j) becomes a separate sequence in the batch
        x = x.reshape(B * H * W, T, 1)     # (B*H*W, T, 1)

        # Project to d_model
        x = self.input_proj(x)             # (B*H*W, T, d_model)

        # Run transformer in chunks to avoid 65535 batch limit (not specific to gpu, is universal)
        total_tokens = x.shape[0]         # this is B*H*W
        if total_tokens > self.max_tokens:
            chunks = []
            for start in range(0, total_tokens, self.max_tokens):
                end = min(start + self.max_tokens, total_tokens)
                x_chunk = x[start:end]            # (chunk_size, T, d_model)
                x_chunk = self.encoder(x_chunk)   # same shape
                chunks.append(x_chunk)
            x = torch.cat(chunks, dim=0)
        else:
            x = self.encoder(x)

        # Project back to scalar
        x = self.output_proj(x)            # (B*H*W, T, 1)

        # Reshape back to (B, T, H, W)
        x = x.reshape(B, H, W, T)          # (B, H, W, T)
        x = x.permute(0, 3, 1, 2)          # (B, T, H, W)

        return x

class TransformerUNet(nn.Module):
    def __init__(self, n_channels, n_classes, use_temporal_transformer=True, init_hid_dim=8, kernel_size=3, pooling='max', bilinear=False, drop_channels=False, p_drop=None):
        super(TransformerUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_hid_dim = init_hid_dim 
        self.bilinear = bilinear
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.drop_channels = drop_channels
        self.p_drop = p_drop

        self.use_temporal_transformer = use_temporal_transformer
        if use_temporal_transformer:
            self.temporal_transformer = TemporalTransformerBlock(
                T=n_channels,       # the 4 time steps (or however many you use)
                d_model=8,         # embedding size; safe default
                nhead=4,            # must divide d_model
                num_layers=2,
                dim_feedforward=64,
                dropout=0.1,
            )

        hid_dims = [init_hid_dim * (2**i) for i in range(5)]
        self.hid_dims = hid_dims

        # initial 2D Convolution
        self.inc = DoubleConv(n_channels, hid_dims[0], kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

        # downscaling with 2D Convolution followed by pooling
        self.down1 = Down(hid_dims[0], hid_dims[1], kernel_size, pooling, drop_channels, p_drop)
        self.down2 = Down(hid_dims[1], hid_dims[2], kernel_size, pooling, drop_channels, p_drop)
        self.down3 = Down(hid_dims[2], hid_dims[3], kernel_size, pooling, drop_channels, p_drop)

        # downscaling with 2D Convolution followed by pooling
        factor = 2 if bilinear else 1
        self.down4 = Down(hid_dims[3], hid_dims[4] // factor, kernel_size, pooling, drop_channels, p_drop)

        # upscaling with 2D Convolution followed by Double Convolution
        self.up1 = Up(hid_dims[4], hid_dims[3] // factor, kernel_size, bilinear, drop_channels, p_drop)
        self.up2 = Up(hid_dims[3], hid_dims[2] // factor, kernel_size, bilinear, drop_channels, p_drop)
        self.up3 = Up(hid_dims[2], hid_dims[1] // factor, kernel_size, bilinear, drop_channels, p_drop)

        # final 2D Convolution for output
        self.up4 = Up(hid_dims[1], hid_dims[0], kernel_size, bilinear, drop_channels, p_drop)
        self.outc = OutConv(hid_dims[0], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T, H, W) where T = n_channels (e.g., 4)
        if self.use_temporal_transformer:
            x = self.temporal_transformer(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        x  = self.outc(x)
        x  = self.sigmoid(x)

        return x
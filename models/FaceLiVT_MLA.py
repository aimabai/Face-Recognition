# models/face_livt_mla.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# -------------------------
# Utility layers
# -------------------------
class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1, act=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups, bias=False),
                  nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)

# -------------------------
# RepMix (training-time) block
# Depthwise 3x3 + depthwise 1x1 + BN; identity branch included.
# -------------------------
class RepMixBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # depthwise branches
        self.dw_k = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn_k = nn.BatchNorm2d(dim)
        self.dw_1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=dim, bias=False)
        self.bn_1 = nn.BatchNorm2d(dim)
        # No extra conv for identity to keep it lightweight (paper uses identity)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out_k = self.bn_k(self.dw_k(x))
        out_1 = self.bn_1(self.dw_1(x))
        out = out_k + out_1
        out = self.act(out)
        return identity + out

# -------------------------
# MHLA approximation that works with variable H, W:
# - For each head, reduce tokens using adaptive pooling to Nr tokens
# - Apply small per-token MLP (channel-wise) and then upsample back to N tokens
# This keeps the spirit of token-reduction (Wi -> nonlinearity -> Wo) while
# supporting variable spatial sizes.
# -------------------------
class MHLA(nn.Module):
    def __init__(self, dim, num_heads=8, token_reduction_ratio=0.25):
        """
        dim: total channels
        num_heads: number of heads (must divide dim)
        token_reduction_ratio: Nr = max(1, int(N * token_reduction_ratio))
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.token_reduction_ratio = token_reduction_ratio

        # small per-head channel projection applied to reduced tokens
        # implement as a shared MLP on token axis (applied per-head)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, self.head_dim)
        )

        # final projection to mix heads back (pointwise conv)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        Ch = self.head_dim
        Nr = max(1, int(N * self.token_reduction_ratio))

        # reshape to tokens per head
        x = x.reshape(B, self.num_heads, Ch, N)   # (B, He, Ch, N)
        out_heads = []
        for h in range(self.num_heads):
            xh = x[:, h, :, :]                     # (B, Ch, N)
            # reduce tokens -> (B, Ch, Nr)
            xh_red = F.adaptive_avg_pool1d(xh, Nr)
            # bring tokens to axis 1 for linear on channel dimension per token
            xh_red_t = xh_red.permute(0, 2, 1)     # (B, Nr, Ch)
            # token_mlp acts on the channel dimension (per reduced token)
            xh_red_t = self.token_mlp(xh_red_t)    # (B, Nr, Ch)
            # upsample tokens back to N
            xh_up = xh_red_t.permute(0, 2, 1)      # (B, Ch, Nr)
            xh_up = F.interpolate(xh_up, size=N, mode='linear', align_corners=False)  # (B, Ch, N)
            out_heads.append(xh_up)

        x_out = torch.cat(out_heads, dim=1)         # (B, C, N)
        x_out = x_out.reshape(B, C, H, W)           # (B, C, H, W)
        # final pointwise projection (mix channels)
        x_out = self.out_proj(x_out)
        return x_out

# -------------------------
# FaceLiVT Block
# Pre-norm for token and channel mixers, with residuals
# -------------------------
class FaceLiVTBlock(nn.Module):
    def __init__(self, dim, use_attention=False, mhla_heads=8, token_reduction_ratio=0.25, mlp_ratio=3):
        super().__init__()
        self.dim = dim
        self.use_attention = use_attention

        # Norms (BatchNorm used in paper)
        self.norm1 = nn.BatchNorm2d(dim)
        if use_attention:
            self.token_mixer = MHLA(dim, num_heads=mhla_heads, token_reduction_ratio=token_reduction_ratio)
        else:
            self.token_mixer = RepMixBlock(dim)

        # MLP (channel mixing) with expansion ratio r
        hidden_dim = dim * mlp_ratio
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        # token mixing
        x = x + self.token_mixer(self.norm1(x))
        # channel mixing
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------
# Main FaceLiVT model (M-LA variant)
# dims & stages chosen to approximate the ≈9.5M parameter model
# stages: [2,2,6,2], dims: [64,128,192,256]
# -------------------------
class FaceLiVT_M_LA(nn.Module):
    def __init__(self, nOut=512, stages=[2,2,6,2],
                 dims=[64,128,192,256],
                 mhla_heads=8, token_reduction_ratio=0.25,
                 img_size=224,
                 pretrained=False,  
                 **kwargs):          

        """
        nOut: output embedding dim
        stages: number of blocks per stage
        dims: channel dims per stage
        mhla_heads: number of heads for MHLA blocks
        token_reduction_ratio: Nr = int(N * ratio) for MHLA
        img_size: input image size (square). Model adapts to other sizes at runtime.
        """
        super().__init__()
        assert len(stages) == 4 and len(dims) == 4

        # Stem: two stride-2 convs
        self.stem = nn.Sequential(
            ConvBNAct(3, dims[0] // 2, kernel=3, stride=2, padding=1),   # 224 -> 112
            ConvBNAct(dims[0] // 2, dims[0], kernel=3, stride=2, padding=1)  # 112 -> 56 (paper uses 112->56->28 for 112 input)
        )

        # Build stages
        self.stage1 = self._make_stage(dims[0], dims[0], stages[0], use_attention=False,
                                       mhla_heads=mhla_heads, token_reduction_ratio=token_reduction_ratio)
        self.stage2 = self._make_stage(dims[0], dims[1], stages[1], use_attention=False,
                                       mhla_heads=mhla_heads, token_reduction_ratio=token_reduction_ratio)
        self.stage3 = self._make_stage(dims[1], dims[2], stages[2], use_attention=True,
                                       mhla_heads=mhla_heads, token_reduction_ratio=token_reduction_ratio)
        self.stage4 = self._make_stage(dims[2], dims[3], stages[3], use_attention=True,
                                       mhla_heads=mhla_heads, token_reduction_ratio=token_reduction_ratio)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dims[3], nOut)

        self._initialize_weights()

    def _make_stage(self, in_dim, out_dim, num_blocks, use_attention=False,
                    mhla_heads=8, token_reduction_ratio=0.25):
        layers = []
        # downsample if dims change (stride 2)
        if in_dim != out_dim:
            layers.append(ConvBNAct(in_dim, out_dim, kernel=3, stride=2, padding=1))
            in_dim = out_dim

        # add blocks
        for _ in range(num_blocks):
            layers.append(FaceLiVTBlock(in_dim, use_attention=use_attention,
                                       mhla_heads=mhla_heads, token_reduction_ratio=token_reduction_ratio))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # follow standard initializations (paper used common conv/Bn/linear inits)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B, 3, H, W) — H & W can vary (we adapted MHLA to support variable sizes)
        x = self.stem(x)      # reduce spatial resolution
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# helper constructor
def MainModel(nOut=512, **kwargs):
    return FaceLiVT_M_LA(nOut=nOut, **kwargs)
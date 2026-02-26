# models/FaceLiVT_MLA_maxcap_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -------------------------
# Utilities
# -------------------------
class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1, act=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups, bias=False),
                  nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)

# -------------------------
# RepMix (training-time) block: depthwise 3x3 + depthwise 1x1 + BN + GELU + identity residual
# -------------------------
class RepMixBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.dw_k = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn_k = nn.BatchNorm2d(dim)
        self.dw_1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=dim, bias=False)
        self.bn_1 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out_k = self.bn_k(self.dw_k(x))
        out_1 = self.bn_1(self.dw_1(x))
        out = out_k + out_1
        out = self.act(out)
        return identity + out

# -------------------------
# MHLA (true learned Wi/Wo per block and per head)
# - For a given block, Wi: (He, N, Nr), Wo: (He, Nr, N)
# - We use einsum for efficient batch/head matmul
# -------------------------
class MHLA_true(nn.Module):
    def __init__(self, dim: int, num_heads: int, N: int, Nr: int):
        """
        dim: total number of channels for this block
        num_heads: number of heads (must divide dim)
        N: number of tokens (H*W) for this stage (fixed for 224 input pyramid)
        Nr: reduced token count (Nr << N)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.N = N
        self.Nr = Nr
        self.ch = dim // num_heads

        # Learned token reduction and expansion matrices per head
        # Wi: (He, N, Nr)
        # Wo: (He, Nr, N)
        self.Wi = nn.Parameter(torch.randn(num_heads, N, Nr) * (1.0 / (N ** 0.5)))
        self.Wo = nn.Parameter(torch.randn(num_heads, Nr, N) * (1.0 / (Nr ** 0.5)))

        # small gating / token-channel mixing handled by a pointwise conv (out projection)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # initialize (small)
        nn.init.normal_(self.Wi, std=0.02)
        nn.init.normal_(self.Wo, std=0.02)
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) where H*W == N
        B, C, H, W = x.shape
        N = H * W
        assert N == self.N, f"Expected N={self.N} tokens for this MHLA module, got {N}. Model assumes fixed spatial sizes per stage for true MHLA."

        # reshape to (B, He, Ch, N)
        x = x.view(B, self.num_heads, self.ch, N)           # (B, He, Ch, N)
        # move token axis forward for matmul
        x_t = x.permute(0, 1, 3, 2).contiguous()           # (B, He, N, Ch)
        # compute reduced tokens per head: (B, He, Nr, Ch) = einsum('bhnc,hnr->bhrc')
        # Note: Wi shape (He, N, Nr)
        red = torch.einsum('bhnc,hnr->bhrc', x_t, self.Wi)  # (B, He, Nr, Ch)
        red = F.gelu(red)
        # expand back: (B, He, N, Ch) = einsum('bhrc,hrn->bhnc')
        expanded = torch.einsum('bhrc,hrn->bhnc', red, self.Wo)  # (B, He, N, Ch)
        # permute back to (B, He, Ch, N)
        out = expanded.permute(0, 1, 3, 2).contiguous()    # (B, He, Ch, N)
        # merge heads to (B, C, N)
        out = out.view(B, C, N)                            # (B, C, N)
        # reshape to (B, C, H, W)
        out = out.view(B, C, H, W)
        # final out projection (mix channels)
        out = self.out_proj(out)
        return out

# -------------------------
# FaceLiVT Block: Pre-norm with BN, token mixer, channel MLP (r=3), residuals
# -------------------------
class FaceLiVTBlock(nn.Module):
    def __init__(self, dim: int, use_attention: bool = False, mhla_cfg: Tuple[int,int,int] = None, mlp_ratio: int = 3):
        """
        mhla_cfg: (num_heads, N, Nr) if use_attention True
        """
        super().__init__()
        self.dim = dim
        self.use_attention = use_attention

        self.norm1 = nn.BatchNorm2d(dim)
        if use_attention:
            assert mhla_cfg is not None, "mhla_cfg must be provided for MHLA blocks"
            num_heads, N, Nr = mhla_cfg
            self.token_mixer = MHLA_true(dim, num_heads=num_heads, N=N, Nr=Nr)
        else:
            self.token_mixer = RepMixBlock(dim)

        self.norm2 = nn.BatchNorm2d(dim)
        hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------
# FaceLiVT main model (Option 3 - near capacity for 224x224) - FIXED HEADS
# -------------------------
class FaceLiVT_M_LA(nn.Module):
    def __init__(self, nOut: int = 512,
                 stages: List[int] = [2, 2, 6, 2],
                 dims: List[int] = [64, 192, 352, 448],
                 mhla_heads_stage3: int = 16,
                 mhla_heads_stage4: int = 32,
                 nr_stage3: int = 12,
                 nr_stage4: int = 16,
                 pretrained: bool = False,
                 img_size: int = 224,
                 **kwargs):
        """
        dims: channel dims per stage
        stages: number of blocks per stage
        mhla_heads_stage3 / stage4: heads for MHLA in stage3 and stage4
        nr_stage3 / stage4: reduced token counts for MHLA (Nr)
        Note: This implementation uses fixed spatial sizes per stage (for true MHLA).
        For 224x224 input and the stem used here:
            after stem: 224 -> 112 -> 56 (stage1 at 56x56)
            stage2 downsample -> 28x28
            stage3 downsample -> 14x14  (N_stage3 = 196)
            stage4 downsample -> 7x7    (N_stage4 = 49)
        """
        super().__init__()
        assert len(stages) == 4 and len(dims) == 4, "stages and dims must be length 4"

        # Stem: two stride-2 convs, 224->112->56
        self.stem = nn.Sequential(
            ConvBNAct(3, dims[0] // 2, kernel=3, stride=2, padding=1),   # 224 -> 112
            ConvBNAct(dims[0] // 2, dims[0], kernel=3, stride=2, padding=1)  # 112 -> 56
        )

        # Build stages
        self.stage1 = self._make_stage(dims[0], dims[0], stages[0], use_attention=False)
        self.stage2 = self._make_stage(dims[0], dims[1], stages[1], use_attention=False)
        # For stage3, MHLA needs N=14*14=196
        N3 = 14 * 14
        self.stage3 = self._make_stage(dims[1], dims[2], stages[2], use_attention=True,
                                       mhla_cfg=(mhla_heads_stage3, N3, nr_stage3))
        # For stage4, MHLA needs N=7*7=49
        N4 = 7 * 7
        self.stage4 = self._make_stage(dims[2], dims[3], stages[3], use_attention=True,
                                       mhla_cfg=(mhla_heads_stage4, N4, nr_stage4))

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dims[3], nOut)

        # allow pretrained kw but ignore (keeps compatibility)
        self.pretrained = pretrained

        self._initialize_weights()

    def _make_stage(self, in_dim: int, out_dim: int, num_blocks: int, use_attention: bool = False, mhla_cfg: Tuple[int,int,int] = None):
        layers = []
        if in_dim != out_dim:
            layers.append(ConvBNAct(in_dim, out_dim, kernel=3, stride=2, padding=1))
            in_dim = out_dim

        for i in range(num_blocks):
            if use_attention:
                # each block gets its own MHLA_true with same mhla_cfg
                layers.append(FaceLiVTBlock(in_dim, use_attention=True, mhla_cfg=mhla_cfg))
            else:
                layers.append(FaceLiVTBlock(in_dim, use_attention=False))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # follow common initializations
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) must be 224x224 for the MHLA shapes used here
        B, C, H, W = x.shape
        if H != 224 or W != 224:
            raise ValueError(f"FaceLiVT_M_LA (true-MHLA mode) expects input size 224x224. Got {H}x{W}.")

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# factory functions for compatibility with existing code
def FaceLiVT_MLA(nOut=512, **kwargs):
    return FaceLiVT_M_LA(nOut=nOut, **kwargs)

def MainModel(nOut=512, **kwargs):
    return FaceLiVT_M_LA(nOut=nOut, **kwargs)

# quick sanity check when running this file directly
if __name__ == "__main__":
    model = FaceLiVT_M_LA(nOut=512,
                          stages=[2,2,6,2],
                          dims=[64,192,352,448],
                          mhla_heads_stage3=16,
                          mhla_heads_stage4=32,
                          nr_stage3=16,
                          nr_stage4=20,
                          pretrained=False)
    x = torch.randn(2,3,224,224)
    y = model(x)
    print("Output shape:", y.shape)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: {:,}".format(total))

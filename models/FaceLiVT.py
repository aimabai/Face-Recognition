# models/FaceLiVT.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """Simplified attention that actually works"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [layer.reshape(B, self.num_heads, self.head_dim, H * W) for layer in qkv]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = torch.matmul(attn, v)
        x = x.reshape(B, C, H, W)
        return self.proj(x)

class RepMixBlock(nn.Module):
    """Working RepMix block"""
    def __init__(self, dim):
        super().__init__()
        self.dw_conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw_conv2 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        x1 = self.dw_conv1(x)
        x1 = self.bn1(x1)
        x2 = self.dw_conv2(x)  
        x2 = self.bn2(x2)
        return identity + self.act(x1 + x2)

class FaceLiVTBlock(nn.Module):
    def __init__(self, dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        if use_attention:
            self.token_mixer = SimpleAttention(dim)
        else:
            self.token_mixer = RepMixBlock(dim)
            
        # MLP
        hidden_dim = dim * 2
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )
        self.norm = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        # Token mixing
        x = x + self.token_mixer(self.norm(x))
        # Channel mixing  
        x = x + self.mlp(self.norm(x))
        return x

class FaceLiVT(nn.Module):
    def __init__(self, nOut=512, stages=[2, 4, 6, 2], dims=[64, 128, 256, 512]):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0]//2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]//2),
            nn.GELU(),
            nn.Conv2d(dims[0]//2, dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )
        
        # Stages
        self.stage1 = self._make_stage(dims[0], dims[0], stages[0], use_attention=False)
        self.stage2 = self._make_stage(dims[0], dims[1], stages[1], use_attention=False)
        self.stage3 = self._make_stage(dims[1], dims[2], stages[2], use_attention=True)
        self.stage4 = self._make_stage(dims[2], dims[3], stages[3], use_attention=True)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[3], nOut)
        
        self._initialize_weights()
        
    def _make_stage(self, in_dim, out_dim, num_blocks, use_attention=False):
        layers = []
        
        # Downsample if changing dimensions
        if in_dim != out_dim:
            layers.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.GELU()
            ))
            in_dim = out_dim
        
        # Add blocks
        for _ in range(num_blocks):
            layers.append(FaceLiVTBlock(in_dim, use_attention=use_attention))
            
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Smaller initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x) 
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        
        x = F.normalize(x, p=2, dim=1)
        return x

def MainModel(nOut=512, **kwargs):
    return FaceLiVT(nOut=nOut, stages=[2, 4, 6, 2], dims=[64, 128, 256, 512])
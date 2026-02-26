# models/GhostFaceNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Ghost module & helpers
# -------------------------

def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=True):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SEBlock(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        hidden = max(1, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_ch, bias=True),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GhostModule(nn.Module):
    """
    Ghost module: generates some channels by cheap operations (depthwise conv)
    and concatenates them to form the output channels.
    """

    def __init__(
        self,
        inp,
        oup,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        relu=True,
    ):
        super().__init__()
        self.oup = oup
        init_channels = int(oup / ratio)
        new_channels = oup - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp,
                init_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                stride=1,
                padding=dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # in some cases oup != x1+x2 channel count due to rounding; slice if needed
        out = out[:, : self.oup, :, :].contiguous()
        return out


class GhostBottleneck(nn.Module):
    """
    Ghost bottleneck with optional SE and stride (from GhostNet family)
    """

    def __init__(self, in_ch, mid_ch, out_ch, dw_kernel=3, stride=1, use_se=False):
        super().__init__()
        self.stride = stride

        # 1) pointwise ghost expansion
        self.ghost1 = GhostModule(in_ch, mid_ch, kernel_size=1, relu=True)

        # 2) depthwise conv for downsample
        self.dw = (
            nn.Sequential(
                nn.Conv2d(
                    mid_ch,
                    mid_ch,
                    dw_kernel,
                    stride=stride,
                    padding=dw_kernel // 2,
                    groups=mid_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(mid_ch),
            )
            if stride > 1
            else nn.Identity()
        )

        # 3) SE
        self.se = SEBlock(mid_ch) if use_se else None

        # 4) ghost projection
        self.ghost2 = GhostModule(mid_ch, out_ch, kernel_size=1, relu=False)

        # shortcut
        if (in_ch == out_ch) and (stride == 1):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    in_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = x
        x = self.ghost1(x)
        x = self.dw(x) if self.stride > 1 else x
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        res = self.shortcut(res)
        return x + res


# -------------------------
# GhostFaceNet-XL backbone (≈10.86M params with nOut=512)
# -------------------------

class GhostFaceNet(nn.Module):
    """
    GhostFaceNet-XL for face recognition (under 11.5M params).
    - Input: (B, 3, 224, 224)
    - Output: (B, nOut) L2-normalized embedding
    """

    def __init__(self, nOut=512, pretrained=False, **kwargs):
        super().__init__()

        # Stem: 224 -> 112
        self.in_ch = 48
        self.conv_stem = conv_bn(
            3, self.in_ch, kernel_size=3, stride=2, padding=1, act=True
        )

        # blocks configuration: (mid_channels, out_channels, repeats, stride, use_se)
        # 112x112 -> 112x112
        # 112x112 -> 56x56
        # 56x56   -> 28x28
        # 28x28   -> 14x14
        # 14x14   -> 7x7
        self.cfg = [
            (96, 96, 2, 1, False),   # stage 1
            (192, 192, 4, 2, False), # stage 2
            (320, 320, 7, 2, True),  # stage 3
            (448, 448, 7, 2, True),  # stage 4
            (640, 768, 5, 2, True),  # stage 5
        ]

        stages = []
        in_ch = self.in_ch
        for mid, outc, reps, stride, use_se in self.cfg:
            blocks = []
            s = stride
            for i in range(reps):
                blocks.append(
                    GhostBottleneck(
                        in_ch,
                        mid,
                        outc,
                        stride=s,
                        use_se=use_se,
                    )
                )
                in_ch = outc
                s = 1
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)

        # final conv head before pooling
        self.conv_head = conv_bn(
            in_ch, 2688, kernel_size=1, stride=1, padding=0, act=True
        )

        # pooling + embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2688, nOut, bias=False),
            nn.BatchNorm1d(nOut, eps=1e-5),
        )

        # init
        self._initialize_weights(pretrained)

    def _initialize_weights(self, pretrained=False):
        # No external pretrained weights (as per project rules)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B,3,224,224)
        x = self.conv_stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_head(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)   # (B, nOut), UNNORMALIZED
        return x      


# factory for compatibility with your EmbedNet pipeline
def MainModel(nOut=512, **kwargs):
    return GhostFaceNet(nOut=nOut, **kwargs)


# quick param check when run directly
if __name__ == "__main__":
    model = MainModel(nOut=512, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters (backbone+embedding): {:,}".format(total))

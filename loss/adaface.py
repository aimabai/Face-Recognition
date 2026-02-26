# loss/adaface.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceCore(nn.Module):
    def __init__(self, dim, num_classes, margin=0.4, scale=64, h=0.333, t_alpha=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.h = h
        self.t_alpha = t_alpha

        self.weight = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer("t", torch.zeros(1))

    def forward(self, emb, label):
        """
        emb: (B, dim) UNNORMALIZED embeddings from backbone
        label: (B,)
        """
        z = emb

        # feature norm for quality
        x_norm = z.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)

        # update EMA of feature norms
        with torch.no_grad():
            mean = x_norm.mean()
            if self.t == 0:
                self.t = mean
            self.t = self.t * (1 - self.t_alpha) + mean * self.t_alpha

        # normalized features and weights for cosine
        x = F.normalize(z, p=2, dim=1)                 # (B, dim)
        W = F.normalize(self.weight, p=2, dim=1)       # (C, dim)

        cos = torch.matmul(x, W.t())                   # (B, C)

        # target cos
        idx = torch.arange(emb.size(0), device=emb.device)
        cos_t = cos[idx, label].view(-1, 1)

        # quality-adaptive margin
        margin_i = self.margin + self.h * (x_norm - self.t)
        margin_i = margin_i.clamp(min=0.0, max=1.0)

        # apply margin to target logit
        cos_m = cos_t - margin_i
        logits = cos.clone()
        logits[idx, label] = cos_m.squeeze(1)

        # scale and CE loss
        logits = logits * self.scale
        loss = F.cross_entropy(logits, label)

        return loss



# ========= Wrapper for EmbedNet =========
class LossFunction(nn.Module):
    """
    This wrapper MUST exist for EmbedNet to work.
    EmbedNet will call: LossFunction(**kwargs)
    So kwargs must include: nClasses, nOut, margin, scale
    """

    def __init__(self, nClasses, nOut, margin, scale, **kwargs):
        super().__init__()

        self.loss = AdaFaceCore(
            dim=nOut,
            num_classes=nClasses,
            margin=margin,
            scale=scale,
            h=0.333,
            t_alpha=0.01
        )

    def forward(self, x, label):
        return self.loss(x, label)

"""
CAFHybridLight model definition with tokenizer blocks and CAF module.

This mirrors the architecture used during training so that we can load a
state_dict checkpoint and run inference inside the Flask app.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FrequencyTokenizer(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        hp = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("hp_kernel", hp.view(1, 1, 3, 3))
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_dim, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hp = F.conv2d(x.mean(1, keepdim=True), self.hp_kernel, padding=1)
        x = self.conv(x + x_hp)
        return x.flatten(2).transpose(1, 2)


class MobileNetTokenizer(nn.Module):
    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", pretrained=pretrained, features_only=True
        )
        channels = self.backbone.feature_info[-1]["num_chs"]
        self.proj = nn.Linear(channels, out_dim)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)[-1]
        tokens = feat.flatten(2).transpose(1, 2)
        return self.proj(tokens)


class MobileViTTokenizer(nn.Module):
    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        model_name = "mobilevit_s"
        try:
            self.vit = timm.create_model(
                model_name, pretrained=pretrained, features_only=True
            )
        except Exception:
            self.vit = timm.create_model(
                "mobilenetv3_small_100", pretrained=pretrained, features_only=True
            )
        channels = self.vit.feature_info[-1]["num_chs"]
        self.proj = nn.Linear(channels, out_dim)
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vit(x)[-1]
        tokens = feat.flatten(2).transpose(1, 2)
        return self.proj(tokens)


class CAF(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.ln_v = nn.LayerNorm(dim)
        self.ln_c = nn.LayerNorm(dim)
        self.cross1 = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.cross2 = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.ffn_v = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.ffn_c = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, v: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v2, _ = self.cross1(self.ln_v(v), self.ln_c(c), self.ln_c(c))
        c2, _ = self.cross2(self.ln_c(c), self.ln_v(v), self.ln_v(v))
        v = v + v2
        c = c + c2
        v = v + self.ffn_v(v)
        c = c + self.ffn_c(c)
        return v, c


class CAFHybridLight(nn.Module):
    def __init__(self, d: int = 128, pretrained_tokenizers: bool = True):
        super().__init__()
        self.tok_cnn = MobileNetTokenizer(out_dim=d, pretrained=pretrained_tokenizers)
        self.tok_freq = FrequencyTokenizer(out_dim=d)
        self.tok_vit = MobileViTTokenizer(out_dim=d, pretrained=pretrained_tokenizers)
        self.caf = CAF(dim=d)
        self.head = nn.Linear(2 * d, 2)

        for param in self.tok_cnn.parameters():
            param.requires_grad = False
        for param in self.tok_freq.parameters():
            param.requires_grad = False
        for param in self.tok_vit.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_tok = self.tok_cnn(x)
        freq_tok = self.tok_freq(x)
        vit_tok = self.tok_vit(x)
        c = torch.cat([cnn_tok, freq_tok], dim=1)
        v = vit_tok
        v, c = self.caf(v, c)
        out = torch.cat([v.mean(1), c.mean(1)], dim=1)
        return self.head(out)


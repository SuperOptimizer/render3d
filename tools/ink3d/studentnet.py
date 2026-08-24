"""Scalable 3D residual U-Net students for ink distillation.

Same family as the teacher (nnU-Net-style residual encoder, plain conv
decoder, single-logit ink head) but with width/depth as free knobs so
models 10x-100x smaller than the 142M-param teacher can be built. All
convs are 3^3, downsampling is stride-2 convs, upsampling is transposed
convs, InstanceNorm + LeakyReLU throughout -- fully convolutional, so a
student trained on 128^3 crops runs on 256^3 chunks unchanged.

Presets (param counts printed at build):
  small ~14M  (~10x smaller than the teacher)
  tiny  ~1.4M (~100x smaller)
"""
import torch
import torch.nn as nn

PRESETS = {
    "small": {"features": [20, 40, 80, 160, 320], "blocks": 2},  # 15.3M, ~9x
    "tiny":  {"features": [8, 16, 32, 64, 128],   "blocks": 1},  # 1.28M, ~111x
}


def _norm_act(ch):
    return nn.Sequential(nn.InstanceNorm3d(ch, affine=True),
                         nn.LeakyReLU(1e-2, inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.na1 = _norm_act(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.na2 = _norm_act(cout)
        self.skip = (nn.Conv3d(cin, cout, 1, stride, bias=False)
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = self.na1(self.conv1(x))
        y = self.conv2(y)
        return self.na2[1](self.na2[0](y) + self.skip(x))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, 1, 1, bias=False)
        self.na = _norm_act(cout)

    def forward(self, x):
        return self.na(self.conv(x))


class StudentNet(nn.Module):
    """config: {"features": [...], "blocks": n, "in_channels": 1}"""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        feats = list(config["features"])
        blocks = int(config.get("blocks", 1))
        cin = int(config.get("in_channels", 1))
        self.enc = nn.ModuleList()
        prev = cin
        for si, f in enumerate(feats):
            stage = [ResBlock(prev, f, stride=1 if si == 0 else 2)]
            stage += [ResBlock(f, f) for _ in range(blocks - 1)]
            self.enc.append(nn.Sequential(*stage))
            prev = f
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for si in range(len(feats) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(feats[si], feats[si - 1], 2, 2))
            self.dec.append(ConvBlock(feats[si - 1] * 2, feats[si - 1]))
        self.head = nn.Conv3d(feats[0], 1, 1)

    def forward(self, x):
        skips = []
        for stage in self.enc:
            x = stage(x)
            skips.append(x)
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            x = dec(torch.cat([x, skips[-2 - i]], dim=1))
        return self.head(x)  # logits


def build(config):
    net = StudentNet(config)
    n = sum(p.numel() for p in net.parameters())
    print(f"studentnet: features {config['features']} blocks "
          f"{config.get('blocks', 1)} -> {n / 1e6:.2f}M params")
    return net


def build_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}, have {sorted(PRESETS)}")
    return build(dict(PRESETS[name], in_channels=1))

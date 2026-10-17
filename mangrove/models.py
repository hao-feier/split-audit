"""Model zoo: baselines + proposed attention-enhanced architecture.

All models: in_ch = len(FEATURE_BANDS) (9), out_ch = n_classes (4 change
classes). Temporal models take (t1, t2) pairs.

Architecture summary (for the paper):
- exp0 baseline: U-Net (ResNet18-free, plain conv encoder)
- exp1: U-Net + CBAM on decoder stages
- exp2: U-Net + CBAM + multi-scale attention (MSA) on the bottleneck
- exp3 (temporal, plan C): shared-encoder TemporalUNet with
  attention-gated temporal fusion, operating on (t1, t2) pairs
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(prefer_cuda=True):
    """Training/inference device. The project runs on a CPU-only box AND on
    Fu4080-server (RTX 4080 Laptop, torch 2.14+cu130 in
    /home/lizhuo/venvs/junzhe-gpu) — so ask the runtime, never assume CUDA."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Attention modules
# --------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sa = self.mlp(self.avg(x))
        sm = self.mlp(self.max(x))
        return x * self.sigmoid(sa + sm)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, groups=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx = x.max(1)[0].unsqueeze(1)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, ch, r=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention(k)

    def forward(self, x):
        return self.sa(self.ca(x))


class MultiScaleAttention(nn.Module):
    """MSA: parallel dilated convs at rates 1/2/4/8 + channel gate."""
    def __init__(self, ch):
        super().__init__()
        red = ch // 4
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, red, 3, padding=d, dilation=d), nn.ReLU(inplace=True))
            for d in (1, 2, 4, 8)
        ])
        self.fuse = nn.Conv2d(red * 4, ch, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = torch.cat([c(x) for c in self.convs], 1)
        return x * self.act(self.fuse(out))


# --------------------------------------------------------------------------
# U-Net (plain conv encoder, CPU-friendly)
# --------------------------------------------------------------------------
def make_norm(ch, kind="bn"):
    """bn = BatchNorm2d (default, all existing runs); gn = GroupNorm(8)
    (control: per-sample stats that never mix t1/t2 within a batch)."""
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(8, ch)
    raise ValueError(f"unknown norm {kind}")


class EncoderBlock(nn.Module):
    def __init__(self, cin, cout, double=True, norm="bn"):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                  make_norm(cout, norm), nn.ReLU(inplace=True)]
        if double:
            layers += [nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                       make_norm(cout, norm), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net with optional CBAM (add_cbam) and MSA bottleneck (add_msa)."""

    def __init__(self, in_ch=9, out_ch=4, base_ch=32, add_cbam=False, add_msa=False,
                 norm="bn"):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.enc1 = EncoderBlock(in_ch, chs[0], norm=norm)
        self.enc2 = EncoderBlock(chs[0], chs[1], norm=norm)
        self.enc3 = EncoderBlock(chs[1], chs[2], norm=norm)
        self.enc4 = EncoderBlock(chs[2], chs[3], norm=norm)
        self.pool = nn.MaxPool2d(2)
        self.cbams = nn.ModuleList([
            CBAM(c) if add_cbam else nn.Identity() for c in chs[:3]
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(chs[3], chs[3], 3, padding=1, bias=False),
            make_norm(chs[3], norm), nn.ReLU(inplace=True),
        )
        self.msa = MultiScaleAttention(chs[3]) if add_msa else nn.Identity()
        self.up4 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec4 = EncoderBlock(chs[2] * 2, chs[2], norm=norm)
        self.up3 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec3 = EncoderBlock(chs[1] * 2, chs[1], norm=norm)
        self.up2 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec2 = EncoderBlock(chs[0] * 2, chs[0], norm=norm)
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        e1 = self.cbams[0](self.enc1(x))
        e2 = self.cbams[1](self.enc2(self.pool(e1)))
        e3 = self.cbams[2](self.enc3(self.pool(e2)))
        e4 = self.enc4(self.pool(e3))
        b = self.msa(self.bottleneck(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        return self.head(d2)


# --------------------------------------------------------------------------
# DeepLabv3+ baseline: hand-assembled (torchvision 0.29 dropped the
# ready-made decoder; DeepLabV3 is now a bare segmentation wrapper).
# --------------------------------------------------------------------------
class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ (Chen et al., CVPRW 2018) with a ResNet18 backbone.

    Dilated encoder (layer3/4 rate 2, kept at 1/16 resolution), ASPP with
    rates (12, 24, 36), decoder fuses layer2 (1/8) and layer1 (1/4) skip
    features. Input (N, in_ch, H, W) -> output (N, out_ch, H, W).
    """

    def __init__(self, in_ch=9, out_ch=4, atrous_rates=(12, 24, 36)):
        super().__init__()
        from torchvision.models import resnet18
        from torchvision.models.segmentation.deeplabv3 import ASPP

        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1, bias=False)
        for layer in ("layer3", "layer4"):
            for block in getattr(base, layer):
                block.conv2.stride = (1, 1)
                block.conv2.dilation = (2, 2)
                block.conv2.padding = (2, 2)  # keep spatial size with rate 2
        self.backbone = base
        self.aspp = ASPP(512, atrous_rates, out_channels=256)

        self.skip_hi = nn.Sequential(  # layer2: 128 ch @ 1/8
            nn.Conv2d(128, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
        )
        self.skip_lo = nn.Sequential(  # layer1: 64 ch @ 1/4
            nn.Conv2d(64, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
        )
        self.dec_hi = nn.Sequential(  # @ 1/8
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.dec_lo = nn.Sequential(  # @ 1/4
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, out_ch, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        l1 = b.layer1(x)                  # 1/4
        l2 = b.layer2(l1)                 # 1/8
        l4 = b.layer4(b.layer3(l2))       # 1/16 (dilated)
        x = self.aspp(l4)
        x = F.interpolate(x, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec_hi(torch.cat([x, self.skip_hi(l2)], dim=1))
        x = F.interpolate(x, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec_lo(torch.cat([x, self.skip_lo(l1)], dim=1))
        x = self.cls(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------
# Temporal U-Net (plan C): shared encoder + attention-gated fusion
# --------------------------------------------------------------------------
class TemporalUNet(nn.Module):
    """Encodes t1 and t2 with a shared U-Net encoder; fusion head predicts change.

    Fusion: gate = sigmoid(conv(cat(e_t1, e_t2, |e_t1-e_t2|, e_t1*e_t2)));
    fused = gate * e_t1 + (1-gate) * e_t2 at each decoder stage.
    """

    def __init__(self, in_ch=9, out_ch=4, base_ch=32, add_cbam=True, norm="bn"):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.enc1 = EncoderBlock(in_ch, chs[0], norm=norm)
        self.enc2 = EncoderBlock(chs[0], chs[1], norm=norm)
        self.enc3 = EncoderBlock(chs[1], chs[2], norm=norm)
        self.enc4 = EncoderBlock(chs[2], chs[3], norm=norm)
        self.pool = nn.MaxPool2d(2)
        self.cbams = nn.ModuleList([
            CBAM(c) if add_cbam else nn.Identity() for c in chs[:3]
        ])
        # per-stage temporal fusion gates
        self.gates = nn.ModuleList([
            nn.Conv2d(c * 4, c, 1) for c in chs
        ])
        self.up4 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec4 = EncoderBlock(chs[2] * 2, chs[2], norm=norm)
        self.up3 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec3 = EncoderBlock(chs[1] * 2, chs[1], norm=norm)
        self.up2 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec2 = EncoderBlock(chs[0] * 2, chs[0], norm=norm)
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def _encode(self, x):
        e1 = self.cbams[0](self.enc1(x))
        e2 = self.cbams[1](self.enc2(self.pool(e1)))
        e3 = self.cbams[2](self.enc3(self.pool(e2)))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def _fuse(self, i, a, b):
        feat = torch.cat([a, b, (a - b).abs(), a * b], 1)
        g = torch.sigmoid(self.gates[i](feat))
        return g * a + (1 - g) * b

    def forward(self, x1, x2):
        e1a, e2a, e3a, e4a = self._encode(x1)
        e1b, e2b, e3b, e4b = self._encode(x2)
        f4 = self._fuse(3, e4a, e4b)
        d4 = self.dec4(torch.cat([self.up4(f4), self._fuse(2, e3a, e3b)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self._fuse(1, e2a, e2b)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self._fuse(0, e1a, e1b)], 1))
        return self.head(d2)


class FCSiam(nn.Module):
    """FC-Siam (Daudt et al. 2018): two identical U-Net-style encoders, one
    per date, whose level-wise features are concatenated and decoded into the
    change map. Adapted to the 9-band feature stack and the 4-class head so
    it trains under the exact recipe of the other baselines."""

    def __init__(self, in_ch=9, out_ch=4, base_ch=32, norm="bn"):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        # two independent branches (faithful FC-Siam: no weight sharing)
        self.enc1a = EncoderBlock(in_ch, chs[0], norm=norm)
        self.enc2a = EncoderBlock(chs[0], chs[1], norm=norm)
        self.enc3a = EncoderBlock(chs[1], chs[2], norm=norm)
        self.enc4a = EncoderBlock(chs[2], chs[3], norm=norm)
        self.enc1b = EncoderBlock(in_ch, chs[0], norm=norm)
        self.enc2b = EncoderBlock(chs[0], chs[1], norm=norm)
        self.enc3b = EncoderBlock(chs[1], chs[2], norm=norm)
        self.enc4b = EncoderBlock(chs[2], chs[3], norm=norm)
        self.pool = nn.MaxPool2d(2)
        # decoder: upsampled fused feature + both branch features at each level
        self.up4 = nn.ConvTranspose2d(chs[3] * 2, chs[2], 2, stride=2)
        self.dec4 = EncoderBlock(chs[2] * 3, chs[2], norm=norm)
        self.up3 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec3 = EncoderBlock(chs[1] * 3, chs[1], norm=norm)
        self.up2 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec2 = EncoderBlock(chs[0] * 3, chs[0], norm=norm)
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def _encode(self, x, sfx):
        e1 = getattr(self, f"enc1{sfx}")(x)
        e2 = getattr(self, f"enc2{sfx}")(self.pool(e1))
        e3 = getattr(self, f"enc3{sfx}")(self.pool(e2))
        e4 = getattr(self, f"enc4{sfx}")(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, x1, x2):
        f1a, f2a, f3a, f4a = self._encode(x1, "a")
        f1b, f2b, f3b, f4b = self._encode(x2, "b")
        d4 = self.dec4(torch.cat([self.up4(torch.cat([f4a, f4b], 1)), f3a, f3b], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), f2a, f2b], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), f1a, f1b], 1))
        return self.head(d2)


# Models that take (t1, t2) pairs instead of the concatenated 27-channel stack.
DUAL_INPUT_MODELS = {"temporal_unet", "temporal_unet_gn", "fc_siam",
                     "changeformer"}

MODEL_REGISTRY = {
    "unet": lambda **kw: UNet(**kw),
    "unet_cbam": lambda **kw: UNet(add_cbam=True, **kw),
    "unet_msa": lambda **kw: UNet(add_msa=True, **kw),
    "unet_cbam_msa": lambda **kw: UNet(add_cbam=True, add_msa=True, **kw),
    "deeplabv3plus": lambda **kw: DeepLabV3Plus(**kw),
    "temporal_unet": lambda **kw: TemporalUNet(**kw),
    # Controls: GroupNorm temporal U-Net (per-sample stats, no
    # t1/t2 mixing in batch stats) and the FC-Siam SOTA baseline.
    "temporal_unet_gn": lambda **kw: TemporalUNet(norm="gn", **kw),
    "fc_siam": lambda **kw: FCSiam(**kw),
    # Transformer-based CD baseline and pretrained-encoder control.
    "changeformer": lambda **kw: ChangeFormer(**kw),
    "deeplabv3plus_pre": lambda **kw: DeepLabV3PlusPre(**kw),
}


# --------------------------------------------------------------------------
# Transformer-based CD baseline (ChangeFormer-style)
# and ImageNet-pretrained DeepLabv3+.
# --------------------------------------------------------------------------
class _TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ChangeFormer(nn.Module):
    """ChangeFormer-style Transformer change detection (Chen et al. 2022),
    implemented at a parameter budget comparable to the CNN baselines here
    (the published model uses a ResNet50 per date, which does not fit the
    12 GB GPU at this input size).

    Each date is patch-embedded (16x16 patches -> 32x32 = 1024 tokens) and
    encoded by 4 independent Transformer blocks (dim 256, 8 heads). The
    change branch runs one Transformer block (dim 512) on the concatenated
    per-date tokens, then a strided-conv decoder upsamples the token grid
    back to full resolution for the four-class head.
    """

    def __init__(self, in_ch=9, out_ch=4, patch=16, grid=32, dim=256,
                 depth=4, heads=8):
        super().__init__()
        self.grid = grid
        self.pe = nn.Parameter(0.02 * torch.randn(1, grid * grid, dim))
        self.embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.enc = nn.ModuleList([_TransformerBlock(dim, heads) for _ in range(depth)])
        self.change_blk = _TransformerBlock(dim * 2, heads * 2)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, out_ch, 1),
        )

    def _encode(self, x):
        # (B, C, H, W) -> (B, L, dim)
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = t + self.pe
        for blk in self.enc:
            t = blk(t)
        return t

    def forward(self, x1, x2):
        t1, t2 = self._encode(x1), self._encode(x2)
        t = self.change_blk(torch.cat([t1, t2], dim=2))
        b, l, c = t.shape
        t = t.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        return self.dec(t)


def _load_safetensors(path):
    """Minimal safetensors reader (no dependency): returns {name: tensor}."""
    import struct
    raw = Path(path).read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + hlen])
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        dtype = meta["dtype"]
        arr = np.frombuffer(raw, dtype=_NP_DTYPE[dtype],
                            offset=8 + hlen + s, count=(e - s) // _NP_DTYPE[dtype].itemsize)
        out[name] = torch.from_numpy(arr.astype(_NP_DTYPE[dtype])).reshape(meta["shape"])
    return out


_NP_DTYPE = {"F32": np.dtype("float32"), "F16": np.dtype("float16"),
             "I32": np.dtype("int32"), "I64": np.dtype("int64"),
             "U8": np.dtype("uint8"), "I8": np.dtype("int8"),
             "I16": np.dtype("int16"), "BOOL": np.dtype("bool")}


def _pretrained_resnet18_path():
    env = os.environ.get("JZ_PRETRAINED_RESNET18")
    if env:
        return _Path(env)
    return Path(__file__).resolve().parent.parent / "data" / "pretrained" / \
        "resnet18_imagenet.safetensors"


class DeepLabV3PlusPre(DeepLabV3Plus):
    """DeepLabv3+ with an ImageNet-pretrained ResNet18 backbone (fine-tuned
    end-to-end; the first conv is re-initialised for the 9-band input)."""

    def __init__(self, in_ch=9, out_ch=4, atrous_rates=(12, 24, 36),
                 weights_path=None):
        super().__init__(in_ch=in_ch, out_ch=out_ch, atrous_rates=atrous_rates)
        wpath = Path(weights_path) if weights_path else _pretrained_resnet18_path()
        if not wpath.exists():
            raise FileNotFoundError(
                f"pretrained ResNet18 weights not found at {wpath} "
                f"(weights live in data/pretrained/)")
        sd = _load_safetensors(wpath)
        model_sd = self.backbone.state_dict()
        load = {k: v for k, v in sd.items()
                if k in model_sd and k not in ("conv1.weight", "fc.weight", "fc.bias")}
        missing, unexpected = self.backbone.load_state_dict(load, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        # conv1 stays at its fresh (kaiming) initialisation for the 9-band input
        assert "conv1.weight" in missing

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def up(x):
    return F.interpolate(x, scale_factor=2, mode="trilinear")


def down(x):
    return F.interpolate(x, scale_factor=0.5, mode="trilinear")


def norm(c, instance):
    return nn.InstanceNorm3d(c, affine=True) if instance else nn.BatchNorm3d(c)


class ConvBNReLU3D(nn.Module):
    def __init__(self, cin, cout, instance=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), norm(cout, instance), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Stage(nn.Module):
    """n conv blocks on the channel-concatenated inputs, with an optional learned 2x upsampler."""

    def __init__(self, cin, cout, n, transpose=False, instance=False):
        super().__init__()
        self.conv3d_layers = nn.Sequential(*[ConvBNReLU3D(cin if i == 0 else cout, cout, instance) for i in range(n)])
        if transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose3d(cout, cout, 2, stride=2), norm(cout, instance), nn.ReLU(inplace=True)
            )

    def forward(self, *xs):
        return self.conv3d_layers(torch.cat(xs, 1))


class UNet3D(nn.Module):
    """3D U-Net, ~350K parameters. half=True stops the decoder at 1/2 resolution, instance=True replaces every
    BatchNorm with an InstanceNorm.

    Module names match the competition checkpoints, which load into the default model.
    """

    def __init__(self, half=False, instance=False, channels=(28, 32, 36), num_classes=C.NUM_CLASSES):
        super().__init__()
        a, b, c = channels
        self.half_res = half
        self.variant = ("half" if half else "full") + ("_in" if instance else "")
        self.register_buffer("D", torch.tensor(0))  # unused, kept so the competition checkpoints load strictly
        self.norm = norm(1, instance)
        self.encoder1 = Stage(1, a, 2, instance=instance)
        self.encoder2 = Stage(a, b, 2, instance=instance)
        self.decoder1 = Stage(b, c, 4, instance=instance)
        self.decoder2 = Stage(b + c, b, 2, transpose=not half, instance=instance)
        self.pre = Stage(b, a, 2, instance=instance)
        self.mask = nn.Conv3d(a, num_classes, 1, bias=False)

    def forward(self, x):
        """(B, D, H, W) -> logits (B, classes, D/s, H/s, W/s), s = 2 if half else 1."""
        e1 = self.encoder1(self.norm(x.unsqueeze(1)))
        e2 = self.encoder2(down(e1))
        d2 = self.decoder2(e2, up(self.decoder1(down(e2))))
        if not self.half_res:
            d2 = self.decoder2.upsample(d2)
        return self.mask(self.pre(d2))


def build_model(variant):
    stride, normalisation = C.VARIANTS[variant]
    return UNet3D(half=stride == 2, instance=normalisation == "instance")


def load_model(path, device="cpu"):
    """Loads a checkpoint into the architecture it was trained with, read from its weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet3D(half="decoder2.upsample.0.weight" not in state, instance="norm.running_mean" not in state)
    model.load_state_dict(state)
    return model.to(device).eval()

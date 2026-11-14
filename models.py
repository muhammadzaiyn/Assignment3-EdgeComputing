"""CIFAR-style ResNet-18 that can be trained in FP32, quantized (PTQ / QAT) and channel-pruned.

Differences from the ImageNet ResNet-18:
  * the stem is a single 3x3 conv with stride 1 and there is no max-pool, because the
    images are only 32x32 and we do not want to throw away resolution at the start;
  * the last layer has 10 outputs.

Quantization-friendly details:
  * QuantStub / DeQuantStub at the input and output (they do nothing until convert()).
  * Every ReLU is its own module, so fusion and observers never share a module.
  * The residual add uses FloatFunctional.add_relu, which becomes a real INT8 add after convert().

Each BasicBlock has its own inner width (mid_channels), which is the width of conv1's
output. Structured pruning shrinks only this width, so the same class rebuilds the pruned model.
"""
import gzip
import io

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

STAGE_WIDTHS = [64, 128, 256, 512]
BLOCKS_PER_STAGE = 2
DEFAULT_MID_CHANNELS = [w for w in STAGE_WIDTHS for _ in range(BLOCKS_PER_STAGE)]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when the shape changes, otherwise an empty Sequential (identity)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

        # add + relu in one op; after convert() this runs as a quantized add
        self.add_relu = FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.shortcut(x)
        return self.add_relu.add_relu(out, identity)

    def fuse(self, is_qat):
        fuse = fuse_modules_qat if is_qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
        if len(self.shortcut) == 2:
            fuse(self.shortcut, [["0", "1"]], inplace=True)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = list(DEFAULT_MID_CHANNELS)
        assert len(mid_channels) == 8, "ResNet-18 has 8 basic blocks"
        self.mid_channels = list(mid_channels)
        self.num_classes = num_classes

        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        blocks = []
        in_ch = 64
        for stage, out_ch in enumerate(STAGE_WIDTHS):
            for b in range(BLOCKS_PER_STAGE):
                stride = 2 if (stage > 0 and b == 0) else 1
                mid = self.mid_channels[stage * BLOCKS_PER_STAGE + b]
                blocks.append(BasicBlock(in_ch, mid, out_ch, stride))
                in_ch = out_ch
        self.layer1 = nn.Sequential(blocks[0], blocks[1])
        self.layer2 = nn.Sequential(blocks[2], blocks[3])
        self.layer3 = nn.Sequential(blocks[4], blocks[5])
        self.layer4 = nn.Sequential(blocks[6], blocks[7])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def blocks(self):
        """The 8 BasicBlocks in order."""
        return [blk for layer in (self.layer1, self.layer2, self.layer3, self.layer4) for blk in layer]

    def config(self):
        return {"arch": "resnet18_cifar", "num_classes": self.num_classes,
                "mid_channels": list(self.mid_channels)}

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.dequant(x)

    def fuse_model(self, is_qat=False):
        """Fuse every Conv+BN(+ReLU) group, including the shortcut Conv+BN.

        For PTQ the model must be in eval mode, for QAT in train mode.
        """
        fuse = fuse_modules_qat if is_qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for blk in self.blocks():
            blk.fuse(is_qat)


def build_model(config=None):
    """Rebuild a ResNet18 from the config dict stored in a checkpoint."""
    config = config or {}
    return ResNet18(num_classes=config.get("num_classes", 10),
                    mid_channels=config.get("mid_channels"))


def save_gzip_checkpoint(src_path, dst_path):
    """Write a gzip copy of a checkpoint. Pruned zeros compress very well, dense weights do not."""
    with open(src_path, "rb") as src, gzip.open(dst_path, "wb", compresslevel=9) as dst:
        dst.write(src.read())


def load_fp32_checkpoint(path, map_location="cpu"):
    """Load a float checkpoint ({'state_dict', 'config', ...}) into a fresh model.

    A path ending in .gz is a gzip-compressed checkpoint (used for the 90 %-sparse model).
    """
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            ckpt = torch.load(io.BytesIO(f.read()), map_location=map_location, weights_only=False)
    else:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt

import torch
import torch.nn as nn
import torchvision
import collections
from models.transformer_module import Transformer
from models.convolution_module import ConvBlock, OutputNet
from torchvision.models.resnet import ResNet, BasicBlock


class ResNet18Trans(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False):
        super().__init__()
        self.scale_factor = 16//dcsize
        self.encoder = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, zero_init_residual=True)

        self.tran_decoder = Transformer(layers=4)
        self.tran_decoder_p2 = OutputNet(dim=512)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        raw_x = self.encoder.conv1(x)
        raw_x = self.encoder.bn1(raw_x)
        raw_x = self.encoder.relu(raw_x)
        raw_x = self.encoder.layer1(raw_x)
        raw_x = self.encoder.layer2(raw_x)
        raw_x = self.encoder.layer3(raw_x)
        raw_x = self.encoder.layer4(raw_x)
        bs, c, h, w = raw_x.shape

        # path-transformer
        x = raw_x.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        x = self.tran_decoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x)

        return y
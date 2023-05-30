import torch
import torch.nn as nn
import torchvision
from models.transformer_module import Transformer
from models.convolution_module import ConvBlock, OutputNet


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.encoder(x)


class GMNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decoder(x)


class VGG16Trans(nn.Module):
    def __init__(self,
                 dcsize,
                 batch_norm=True,
                 load_weights=False,
                 dropout_rate=0.1,
                 pretrained=True):

        super().__init__()

        self.scale_factor = 16 // dcsize

        self.encoder = ResNet50Encoder(pretrained)

        self.tran_decoder = Transformer(layers=4, dim=2048)

        self.decoder = GMNDecoder(in_channels=2048,
                                     out_channels=512)

        self.tran_decoder_p2 = OutputNet(dim=512)

    def forward(self, x):
        raw_x = self.encoder(x)
        bs, c, h, w = raw_x.shape

        # path-transformer
        x = raw_x.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        x = self.tran_decoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c, h, w)

        x = self.decoder(x)

        y = self.tran_decoder_p2(x)

        return y

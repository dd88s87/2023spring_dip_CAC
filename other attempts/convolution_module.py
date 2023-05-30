from torch import nn

class ConvBlock(nn.Module):
    """
    Normal Conv Block with BN & ReLU
    """

    def __init__(self, cin, cout, k_size=3, d_rate=1, batch_norm=True, res_link=False):
        super().__init__()
        self.res_link = res_link
        if batch_norm:
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, k_size, padding=d_rate, dilation=d_rate),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, k_size, padding=d_rate, dilation=d_rate),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.res_link:
            return x + self.body(x)
        else:
            return self.body(x)


class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f = self.conv(x)
        f = self.relu(f)
        f = self.conv2(f)
        f = self.sigmoid(f)
        return f * x


class OutputNet(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = ConvBlock(dim, 256, 3)
        self.attention1 = Attention(256)
        self.conv2 = ConvBlock(256, 128, 3)
        self.attention2 = Attention(128)
        self.conv3 = ConvBlock(128, 64, 3)
        self.attention3 = Attention(64)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
		nn.ReLU(True),
    )

def forward(self, x):
    x = self.conv1(x)
    x = self.attention1(x)
    x = self.conv2(x)
    x = self.attention2(x)
    x = self.conv3(x)
    x = self.attention3(x)
    x = self.conv4(x)
    return x
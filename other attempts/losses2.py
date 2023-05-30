import torch
import torch.nn as nn
from einops import rearrange

class DownMSELossWithAdv(nn.Module):
    def __init__(self, size=8, alpha=0.01):
        super().__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.mse = nn.MSELoss(reduction='sum')
        self.alpha = alpha
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * (size//4) * (size//4), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, dmap, gt_density):
        gt_density = self.avgpooling(gt_density) * self.tot
        b, c, h, w = dmap.size()
        assert gt_density.size() == dmap.size()

        mse_loss = self.mse(dmap, gt_density)

        with torch.no_grad():
            fake_density = dmap.detach()
            fake_density.requires_grad = True
            adv_output = self.discriminator(fake_density)
            target = torch.ones((b, 1)).to(dmap.device)
            adv_loss = nn.BCELoss()(adv_output, target)

        total_loss = mse_loss + self.alpha * adv_loss

        return total_loss
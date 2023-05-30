import torch
import torch.nn as nn
from einops import rearrange

class DownMSELossWithSimilarity(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.mse = nn.MSELoss(reduction='sum')
        
    def cosine_similarity(self, x1, x2):
        dot_product = torch.sum(torch.multiply(x1, x2), dim=[1,2,3])
        magnitude_x1 = torch.sqrt(torch.sum(torch.square(x1), dim=[1,2,3]))
        magnitude_x2 = torch.sqrt(torch.sum(torch.square(x2), dim=[1,2,3]))
        return dot_product / (magnitude_x1 * magnitude_x2)

    def forward(self, dmap, gt_density):
        gt_density = self.avgpooling(gt_density) * self.tot
        b, c, h, w = dmap.size()
        assert gt_density.size() == dmap.size()

        mse_loss = self.mse(dmap, gt_density)

        cos_sim = self.cosine_similarity(dmap, gt_density)
        
        total_loss = mse_loss - cos_sim

        return total_loss

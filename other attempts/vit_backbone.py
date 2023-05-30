import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg

class VitBackbone(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x):
        x = self.forward(x)
        return x

def build_vit_backbone(IMAGE_SIZE,PATCH_SIZE,EMBED_DIM,DEPTH,NUM_HEADS):
    model = VitBackbone(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=3,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )
    return model
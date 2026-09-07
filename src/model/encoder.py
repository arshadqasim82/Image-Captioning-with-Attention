import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(0, 2, 1)

        return features

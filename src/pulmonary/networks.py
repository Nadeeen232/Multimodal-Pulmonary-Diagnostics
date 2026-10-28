"""Torch models with explicit checkpoint-compatible feature extraction."""

import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, name="resnet50", pretrained=True, dropout=0.5):
        super().__init__()
        if pretrained:
            os.environ.setdefault("TORCH_HOME", str(Path("artifacts/torch-cache").resolve()))
        self.name = name
        if name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.feature_dim = net.fc.in_features
        elif name == "densenet121":
            net = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
            )
            self.features = nn.Sequential(net.features, nn.ReLU(inplace=False))
            self.feature_dim = net.classifier.in_features
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 2))

    def embed(self, images):
        return self.pool(self.features(images)).flatten(1)

    def forward(self, images):
        return self.head(self.embed(images))


class FusionHead(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )

    def forward(self, features):
        return self.layers(features)


def load_encoder(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ImageEncoder(checkpoint["backbone"], pretrained=False, dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(
	num_classes: int,
	pretrained: bool = True,
	freeze_backbone: bool = False,
) -> nn.Module:
	weights: Optional[ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
	model = resnet18(weights=weights)
	model.fc = nn.Linear(model.fc.in_features, num_classes)

	if freeze_backbone:
		for name, param in model.named_parameters():
			if not name.startswith("fc."):
				param.requires_grad = False

	return model

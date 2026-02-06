from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from .model import build_model
from .utils import get_device


def _val_transform() -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict a single image")
	parser.add_argument("--ckpt", default="checkpoints/best.pt")
	parser.add_argument("--image", required=True)
	args = parser.parse_args()

	if not os.path.exists(args.image):
		print(f"Image not found: {args.image}")
		return

	ckpt = torch.load(args.ckpt, map_location="cpu")
	class_names = ckpt.get("class_names")
	if not class_names:
		print("Checkpoint missing class names.")
		return

	device = get_device()
	model = build_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(device)
	model.eval()

	image = Image.open(args.image).convert("RGB")
	x = _val_transform()(image).unsqueeze(0).to(device)

	with torch.no_grad():
		logits = model(x)
		probs = torch.softmax(logits, dim=1).squeeze(0)
		top5_probs, top5_idx = torch.topk(probs, k=5)

	top1_idx = top5_idx[0].item()
	top1_prob = top5_probs[0].item()
	print(f"Top-1: {class_names[top1_idx]} ({top1_prob:.4f})")

	print("Top-5:")
	for prob, idx in zip(top5_probs.tolist(), top5_idx.tolist()):
		print(f"  {class_names[idx]} ({prob:.4f})")


if __name__ == "__main__":
	main()

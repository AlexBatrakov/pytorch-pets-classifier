from __future__ import annotations

import argparse
import os

from .inference import Predictor


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict a single image")
	parser.add_argument("--ckpt", default="checkpoints/best.pt")
	parser.add_argument("--image", required=True)
	parser.add_argument("--top-k", type=int, default=5)
	args = parser.parse_args()

	if not os.path.exists(args.image):
		print(f"Image not found: {args.image}")
		return

	try:
		predictor = Predictor.from_checkpoint(args.ckpt)
		result = predictor.predict_path(args.image, top_k=args.top_k)
	except ValueError as exc:
		print(str(exc))
		return

	print(f"Top-1: {result.label} ({result.score:.4f})")

	print(f"Top-{len(result.top_k)}:")
	for prediction in result.top_k:
		print(f"  {prediction.label} ({prediction.score:.4f})")


if __name__ == "__main__":
	main()

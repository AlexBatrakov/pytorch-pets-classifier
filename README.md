# PyTorch Pets Classifier

Minimal, production-style baseline for multi-class image classification on the Oxford-IIIT Pets dataset (37 breeds). Uses transfer learning with torchvision ResNet18 ImageNet weights and runs on macOS MPS or CPU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --config configs/default.yaml
```

Common overrides:

```bash
python src/train.py --epochs 10 --batch-size 64 --lr 3e-4 --freeze-epochs 2 --num-workers 0
```

Best checkpoint is saved to `./checkpoints/best.pt`.

## Evaluate

```bash
python src/eval.py --ckpt checkpoints/best.pt
```

## Predict

```bash
python src/predict.py --ckpt checkpoints/best.pt --image path/to/image.jpg
```

Example output:

```
Top-1: abyssinian (0.9234)
Top-5:
	abyssinian (0.9234)
	bengal (0.0345)
	siamese (0.0121)
	ragdoll (0.0098)
	birman (0.0076)
```

## macOS MPS

The code automatically selects MPS if available via `torch.backends.mps.is_available()`. If MPS is not available, it falls back to CPU.

## Repo hygiene

- Dataset downloads to ./data (not committed)
- Checkpoints saved to ./checkpoints (not committed)

## Roadmap

- Grad-CAM visualization
- AMP training
- Optuna hyperparameter search
- Weights & Biases logging
- ONNX export
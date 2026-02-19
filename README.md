# PyTorch Pets Classifier

Minimal, production-style baseline for multi-class image classification on the Oxford-IIIT Pets dataset (37 breeds). Uses transfer learning with torchvision ResNet18 ImageNet weights and runs on macOS MPS or CPU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install development dependencies (tests):

```bash
pip install -r requirements-dev.txt
```

## Testing

```bash
python -m pytest -q
```

## Train

```bash
python -m src.train --config configs/default.yaml
```

Common overrides:

```bash
python -m src.train --epochs 10 --batch-size 64 --lr 3e-4 --freeze-epochs 2 --num-workers 0
```

Best checkpoint is saved to `./checkpoints/best.pt`.

## Evaluate

```bash
python -m src.eval --ckpt checkpoints/best.pt --split val
```

Evaluate on the official test split:

```bash
python -m src.eval --ckpt checkpoints/best.pt --split test
```

Save a confusion matrix image:

```bash
python -m src.eval --ckpt checkpoints/best.pt --split test --cm-out assets/confusion_matrix.png --cm-normalize
```

## Results

| Split | acc@1 | acc@5 |
| --- | --- | --- |
| Val | 0.832 | 0.988 |
| Test | 0.805 | 0.980 |

![Confusion matrix](assets/confusion_matrix.png)

## Predict

```bash
python -m src.predict --ckpt checkpoints/best.pt --image path/to/image.jpg
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

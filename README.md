# PyTorch Pets Classifier

A PyTorch-based transfer learning image classifier for the Oxford-IIIT Pets dataset (37 pet breeds).

## Overview

This project implements a transfer learning approach for classifying 37 different pet breeds from the Oxford-IIIT Pets dataset. The classifier leverages pre-trained models (e.g., ResNet, EfficientNet) fine-tuned on the pets dataset.

### Features

- 🐕 37-class pet breed classification
- 🔄 Transfer learning with popular architectures (ResNet, EfficientNet, etc.)
- ⚙️ YAML-based configuration system
- 📊 TensorBoard integration for experiment tracking
- 🧪 Comprehensive test suite
- 📦 Modular and extensible codebase

## Dataset

The [Oxford-IIIT Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) contains:
- 37 pet breeds (25 dog breeds and 12 cat breeds)
- ~200 images per class
- ~7,400 total images
- Variations in scale, pose, and lighting conditions

## Project Structure

```
pytorch-pets-classifier/
├── configs/              # Configuration files
│   └── default.yaml     # Default training configuration
├── src/                 # Source code
│   ├── models/         # Model architectures (to be implemented)
│   ├── data/           # Dataset loaders and transforms (to be implemented)
│   ├── training/       # Training loops and utilities (to be implemented)
│   └── utils/          # Helper functions (to be implemented)
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AlexBatrakov/pytorch-pets-classifier.git
cd pytorch-pets-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit `configs/default.yaml` to customize:
- Model architecture (ResNet, EfficientNet, etc.)
- Training hyperparameters (learning rate, batch size, epochs)
- Dataset paths and preprocessing
- Hardware settings (CUDA, seeds)

### Training (To Be Implemented)

```bash
# Train with default configuration
python src/train.py --config configs/default.yaml

# Train with custom configuration
python src/train.py --config configs/custom.yaml

# Resume from checkpoint
python src/train.py --config configs/default.yaml --resume checkpoints/best_model.pth
```

### Evaluation (To Be Implemented)

```bash
# Evaluate on test set
python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth

# Run inference on a single image
python src/predict.py --image path/to/pet/image.jpg --checkpoint checkpoints/best_model.pth
```

### Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir logs/
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Code Structure Guidelines

- Keep model definitions in `src/models/`
- Data loading and augmentation in `src/data/`
- Training logic in `src/training/`
- Utility functions in `src/utils/`

## Configuration Options

Key parameters in `configs/default.yaml`:

- **dataset.num_classes**: Number of pet breeds (37)
- **dataset.batch_size**: Training batch size
- **model.architecture**: Pre-trained model to use
- **model.pretrained**: Whether to use ImageNet pre-trained weights
- **training.epochs**: Number of training epochs
- **training.learning_rate**: Initial learning rate
- **device.use_cuda**: Enable GPU training

## Future Enhancements

- [ ] Implement data loading and preprocessing pipeline
- [ ] Add model architectures and transfer learning logic
- [ ] Implement training loop with validation
- [ ] Add evaluation metrics (accuracy, F1-score, confusion matrix)
- [ ] Implement inference script for single images
- [ ] Add data augmentation strategies
- [ ] Implement ensemble methods
- [ ] Add model export (ONNX, TorchScript)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Oxford-IIIT Pets dataset: [Parkhi et al., 2012](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- PyTorch and torchvision teams for the excellent deep learning framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue on GitHub.
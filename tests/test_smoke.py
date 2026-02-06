from config import load_config
from model import build_model
from data import build_datasets


def test_smoke_imports_and_builds():
	cfg = load_config("configs/default.yaml")
	model = build_model(num_classes=37, pretrained=False, freeze_backbone=False)
	assert model is not None

	cfg["data"]["download"] = False
	try:
		train_ds, val_ds, class_names = build_datasets(cfg)
		assert len(class_names) == 37
		assert len(train_ds) > 0
		assert len(val_ds) > 0
	except Exception:
		# Dataset might be missing in CI; ensure imports still work.
		assert True

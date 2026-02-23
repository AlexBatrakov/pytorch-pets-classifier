from __future__ import annotations

from torchvision import transforms

from src.predict import _predict_transform, _resolve_preprocess_sizes


def _size_tuple(value) -> tuple[int, int]:
	if isinstance(value, int):
		return (value, value)
	return tuple(value)


def test_predict_preprocess_defaults_match_legacy_behavior() -> None:
	image_size, eval_resize_size = _resolve_preprocess_sizes({})
	assert image_size == 224
	assert eval_resize_size == 256

	tf = _predict_transform(image_size=image_size, eval_resize_size=eval_resize_size)
	assert isinstance(tf.transforms[0], transforms.Resize)
	assert _size_tuple(tf.transforms[0].size) == (256, 256)
	assert isinstance(tf.transforms[1], transforms.CenterCrop)
	assert _size_tuple(tf.transforms[1].size) == (224, 224)


def test_predict_preprocess_uses_sizes_from_checkpoint_config() -> None:
	ckpt = {"config": {"data": {"image_size": 256, "eval_resize_size": 292}}}
	image_size, eval_resize_size = _resolve_preprocess_sizes(ckpt)
	assert image_size == 256
	assert eval_resize_size == 292

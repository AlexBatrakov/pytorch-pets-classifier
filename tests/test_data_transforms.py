from __future__ import annotations

from torchvision import transforms

from src.data import _get_image_sizes, _train_transform, _val_transform


def _size_tuple(value) -> tuple[int, int]:
	if isinstance(value, int):
		return (value, value)
	return tuple(value)


def test_image_size_defaults_keep_existing_behavior() -> None:
	image_size, eval_resize_size = _get_image_sizes({})
	assert image_size == 224
	assert eval_resize_size == 256

	train_tf = _train_transform()
	val_tf = _val_transform()

	assert isinstance(train_tf.transforms[0], transforms.RandomResizedCrop)
	assert _size_tuple(train_tf.transforms[0].size) == (224, 224)

	assert isinstance(val_tf.transforms[0], transforms.Resize)
	assert _size_tuple(val_tf.transforms[0].size) == (256, 256)
	assert isinstance(val_tf.transforms[1], transforms.CenterCrop)
	assert _size_tuple(val_tf.transforms[1].size) == (224, 224)


def test_image_size_can_be_parameterized_for_resolution_sweep() -> None:
	image_size, eval_resize_size = _get_image_sizes(
		{"image_size": 320, "eval_resize_size": 384}
	)
	assert image_size == 320
	assert eval_resize_size == 384

	val_tf = _val_transform(image_size=image_size, eval_resize_size=eval_resize_size)
	assert _size_tuple(val_tf.transforms[0].size) == (384, 384)
	assert _size_tuple(val_tf.transforms[1].size) == (320, 320)

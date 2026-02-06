from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}
	return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
	"""Apply flat or dotted-key overrides into a nested dict."""
	new_cfg = deepcopy(cfg)
	for key, value in overrides.items():
		if value is None:
			continue
		if "." in key:
			parts = key.split(".")
			node = new_cfg
			for p in parts[:-1]:
				if p not in node or not isinstance(node[p], dict):
					node[p] = {}
				node = node[p]
			node[parts[-1]] = value
		else:
			new_cfg[key] = value
	return new_cfg

"""Checkpoint and artifact IO within the configured release workspace."""

import json
from pathlib import Path

import torch

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def output_path(path):
    """Resolve a path within the package's outputs directory."""
    path = Path(path).resolve()
    if not path.is_relative_to((RELEASE_ROOT / "outputs").resolve()):
        raise ValueError(f"Output must be inside {RELEASE_ROOT / 'outputs'}: {path}")
    if not path.is_relative_to(RELEASE_ROOT):
        raise ValueError("The outputs directory must not point outside this release")
    return path


def load_checkpoint(path, map_location="cpu", **kwargs):
    # Only use with checkpoints/caches produced locally by this package.
    return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

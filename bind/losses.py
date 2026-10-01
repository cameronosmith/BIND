"""BIND training loss: per-voxel volume CE (target = nearest voxel to GT-XYZ) + grip + rot CE."""
from .model import multiview_losses  # noqa: F401

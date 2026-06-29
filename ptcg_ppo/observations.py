from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
from tcg_rl.encoding import EncoderConfig, encode_observation_numpy


def encode_batch(observations: Sequence[dict], config: EncoderConfig, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("Cannot encode an empty observation batch")
    rows = [encode_observation_numpy(obs, config) for obs in observations]
    keys = rows[0].keys()
    out = {}
    for key in keys:
        array = np.stack([row[key] for row in rows], axis=0)
        out[key] = torch.as_tensor(array, device=device)
    return out


def batch_to_device(batch: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

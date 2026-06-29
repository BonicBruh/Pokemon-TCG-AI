from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from tcg_rl.encoding import EncoderConfig
from tcg_rl.network.actor_critic import ActorCritic
from .metadata import cg_metadata_tensors

DEFAULT_MODEL_CONFIG = {
    "max_options": 128,
    "hidden_size": 128,
    "option_feature_size": 14,
    "card_vocab_size": 4096,
    "attack_vocab_size": 4096,
    "energy_vocab_size": 32,
    "enum_vocab_size": 256,
    "embedding_size": 48,
    "policy_head_mode": "active",
}


def build_model(model_config: dict | None = None, device: str | torch.device = "cpu") -> ActorCritic:
    config = {**DEFAULT_MODEL_CONFIG, **(model_config or {})}
    metadata = cg_metadata_tensors(config["card_vocab_size"], config["attack_vocab_size"])
    model = ActorCritic(**config, **metadata)
    return model.to(device)


def save_checkpoint(path, model, optimizer=None, *, encoder_config: EncoderConfig, model_config: dict, step: int, extra=None):
    payload = {
        "format": "ptcg-ppo-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "encoder_config": asdict(encoder_config),
        "model_config": dict(model_config),
        "step": int(step),
        "extra": extra or {},
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu", load_optimizer_into=None):
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "ptcg-ppo-v1":
        raise ValueError("Unsupported checkpoint format")
    model = build_model(payload["model_config"], device=device)
    model.load_state_dict(payload["model_state_dict"])
    if load_optimizer_into is not None and payload.get("optimizer_state_dict") is not None:
        load_optimizer_into.load_state_dict(payload["optimizer_state_dict"])
    encoder_config = EncoderConfig(**payload["encoder_config"])
    return model, encoder_config, payload

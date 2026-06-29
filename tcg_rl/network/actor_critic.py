"""Modular actor-critic network for padded CABT-style action spaces."""

from __future__ import annotations

import torch

from tcg_rl.network.encoder import ObservationEncoder
from tcg_rl.network.heads import MultiSelectPolicyHead, ValueHead


class ActorCritic(torch.nn.Module):
    """Actor-critic assembled from encoder, policy head, and value head."""

    def __init__(
        self,
        max_options: int,
        hidden_size: int = 64,
        option_feature_size: int = 14,
        *,
        card_vocab_size: int = 2048,
        attack_vocab_size: int = 2048,
        energy_vocab_size: int = 32,
        enum_vocab_size: int = 256,
        embedding_size: int | None = None,
        policy_head_mode: str = "active",
        card_features: torch.Tensor | None = None,
        attack_features: torch.Tensor | None = None,
        attack_cost_energy_types: torch.Tensor | None = None,
        attack_cost_counts: torch.Tensor | None = None,
    ):
        super().__init__()
        self.max_options = max_options
        self.option_feature_size = option_feature_size
        self.encoder = ObservationEncoder(
            hidden_size=hidden_size,
            option_feature_size=option_feature_size,
            max_options=max_options,
            card_vocab_size=card_vocab_size,
            attack_vocab_size=attack_vocab_size,
            energy_vocab_size=energy_vocab_size,
            enum_vocab_size=enum_vocab_size,
            embedding_size=embedding_size,
            card_features=card_features,
            attack_features=attack_features,
            attack_cost_energy_types=attack_cost_energy_types,
            attack_cost_counts=attack_cost_counts,
        )
        self.policy_head = MultiSelectPolicyHead(hidden_size, max_options, mode=policy_head_mode)
        self.value_head = ValueHead(hidden_size)

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(obs)
        policy = self.policy_head(encoded["board"], encoded["options"])
        return {
            "option_logits": policy["option_logits"],
            "count_logits": policy["count_logits"],
            "values": self.value_head(encoded["board"]),
        }


__all__ = ["ActorCritic"]

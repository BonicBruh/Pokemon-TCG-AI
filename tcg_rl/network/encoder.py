"""Observation encoders for actor-critic policies."""

from __future__ import annotations

import torch


SCALAR_FEATURES = 14
ZONE_FEATURES = 28
ACTIVE_CARD_ZONE_INDICES = (5, 19)
CURRENT_PLAYER_SCALAR_INDEX = 2
CARD_ROLE_HAND = 1
CARD_ROLE_OWN_ACTIVE = 2
CARD_ROLE_OPPONENT_ACTIVE = 3
CARD_ROLE_OWN_BENCH = 4
CARD_ROLE_OPPONENT_BENCH = 5
CARD_ROLE_ACTIVE_TOOL = 6
CARD_ROLE_STADIUM = 7
CARD_ROLE_OPTION = 8
CARD_ROLE_COUNT = 16


class ObservationEncoder(torch.nn.Module):
    """Embed fixed-shape observations into board and action features."""

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        option_feature_size: int = 14,
        max_options: int = 128,
        card_vocab_size: int = 2048,
        attack_vocab_size: int = 2048,
        energy_vocab_size: int = 32,
        enum_vocab_size: int = 256,
        embedding_size: int | None = None,
        card_features: torch.Tensor | None = None,
        attack_features: torch.Tensor | None = None,
        attack_cost_energy_types: torch.Tensor | None = None,
        attack_cost_counts: torch.Tensor | None = None,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if option_feature_size <= 0:
            raise ValueError("option_feature_size must be positive.")
        if max_options <= 0:
            raise ValueError("max_options must be positive.")
        self.hidden_size = hidden_size
        self.option_feature_size = option_feature_size
        self.max_options = max_options
        embed_size = embedding_size or max(8, min(64, hidden_size // 2))
        if embed_size <= 0:
            raise ValueError("embedding_size must be positive.")
        self.embedding_size = embed_size

        self.card_embedding = _offset_embedding(card_vocab_size, embed_size)
        self.attack_embedding = _offset_embedding(attack_vocab_size, embed_size)
        self.energy_embedding = _offset_embedding(energy_vocab_size, embed_size)
        self.enum_embedding = _offset_embedding(enum_vocab_size, embed_size)
        self.card_role_embedding = torch.nn.Embedding(CARD_ROLE_COUNT, embed_size, padding_idx=0)
        self.card_metadata_projection = torch.nn.Sequential(
            torch.nn.Linear(14, embed_size),
            torch.nn.LayerNorm(embed_size),
            torch.nn.Tanh(),
        )
        self.attack_metadata_projection = torch.nn.Sequential(
            torch.nn.Linear(7 + embed_size, embed_size),
            torch.nn.LayerNorm(embed_size),
            torch.nn.Tanh(),
        )
        self.register_buffer(
            "card_features",
            _metadata_buffer(card_features, card_vocab_size + 1, 14),
            persistent=False,
        )
        self.register_buffer(
            "attack_features",
            _metadata_buffer(attack_features, attack_vocab_size + 1, 6),
            persistent=False,
        )
        self.register_buffer(
            "attack_cost_energy_types",
            _metadata_buffer(attack_cost_energy_types, attack_vocab_size + 1, 8, fill=-1),
            persistent=False,
        )
        self.register_buffer(
            "attack_cost_counts",
            _metadata_vector_buffer(attack_cost_counts, attack_vocab_size + 1),
            persistent=False,
        )

        self.numeric_trunk = torch.nn.Sequential(
            torch.nn.Linear(SCALAR_FEATURES + ZONE_FEATURES, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Tanh(),
        )
        board_parts = hidden_size + embed_size * 6 + hidden_size * 2
        self.bench_trunk = torch.nn.Sequential(
            torch.nn.Linear(embed_size + 3, hidden_size),
            torch.nn.Tanh(),
        )
        self.board_projection = torch.nn.Sequential(
            torch.nn.Linear(board_parts, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Tanh(),
        )
        option_categorical_indices = {0, 2, 4, 8, 9, 10, 11, 13}
        option_parts = embed_size * len(option_categorical_indices) + (option_feature_size - len(option_categorical_indices))
        self.option_projection = torch.nn.Sequential(
            torch.nn.Linear(option_parts, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Tanh(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scalar = _require_last_dim(obs, "scalar", SCALAR_FEATURES).to(dtype=torch.float32)
        zones = _require_last_dim(obs, "zones", ZONE_FEATURES).to(dtype=torch.float32)
        if "option_features" not in obs:
            raise ValueError("obs is missing required tensor 'option_features'.")
        option_features = obs["option_features"]
        if option_features.shape[-1:] != (self.option_feature_size,):
            raise ValueError("obs['option_features'] feature size must match the model option_feature_size.")
        if option_features.ndim != scalar.ndim + 1:
            raise ValueError(
                "obs['option_features'] must have shape [..., max_options, option_feature_size]."
            )
        if option_features.shape[:-2] != scalar.shape[:-1]:
            raise ValueError("obs['option_features'] batch shape must match scalar/zones batch shape.")
        if option_features.shape[-2] != self.max_options:
            raise ValueError("obs['option_features'] max_options must match the model max_options.")
        if zones.shape[:-1] != scalar.shape[:-1]:
            raise ValueError("obs['zones'] batch shape must match obs['scalar'].")

        numeric = torch.cat((scalar / 100.0, zones / 100.0), dim=-1)
        numeric_embed = self.numeric_trunk(numeric)
        board_embed = self.board_projection(
            torch.cat(
                (
                    numeric_embed,
                    *self._active_card_embeds(scalar, zones),
                    self._card_set_embed(obs, "hand_card_ids", scalar.shape[:-1], role=CARD_ROLE_HAND),
                    self._bench_embed(obs, "bench", scalar.shape[:-1], role=CARD_ROLE_OWN_BENCH),
                    self._bench_embed(obs, "opponent_bench", scalar.shape[:-1], role=CARD_ROLE_OPPONENT_BENCH),
                    self._energy_set_embed(obs, "active_energy_types", scalar.shape[:-1]),
                    self._card_set_embed(obs, "active_tool_card_ids", scalar.shape[:-1], role=CARD_ROLE_ACTIVE_TOOL),
                    self._stadium_card_embed(obs, scalar.shape[:-1]),
                ),
                dim=-1,
            )
        )
        option_embed = self._option_embed(option_features)
        return {"board": board_embed, "options": option_embed}

    def _active_card_embeds(self, scalar: torch.Tensor, zones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        player0 = zones[..., ACTIVE_CARD_ZONE_INDICES[0]].to(dtype=torch.long)
        player1 = zones[..., ACTIVE_CARD_ZONE_INDICES[1]].to(dtype=torch.long)
        current_player = scalar[..., CURRENT_PLAYER_SCALAR_INDEX].to(dtype=torch.long)
        own_ids = torch.where(current_player.eq(0), player0, player1)
        opponent_ids = torch.where(current_player.eq(0), player1, player0)
        return (
            self._card_embed(own_ids, role=CARD_ROLE_OWN_ACTIVE),
            self._card_embed(opponent_ids, role=CARD_ROLE_OPPONENT_ACTIVE),
        )

    def _card_set_embed(self, obs: dict[str, torch.Tensor], key: str, batch_shape: torch.Size, *, role: int) -> torch.Tensor:
        ids = _require_batch_prefix(obs, key, batch_shape).to(dtype=torch.long)
        ids = ids.reshape(*batch_shape, -1)
        return self._masked_mean_card_embedding(ids, role=role)

    def _energy_set_embed(self, obs: dict[str, torch.Tensor], key: str, batch_shape: torch.Size) -> torch.Tensor:
        ids = _require_batch_prefix(obs, key, batch_shape).to(dtype=torch.long)
        ids = ids.reshape(*batch_shape, -1)
        return _masked_mean_embedding(self.energy_embedding, ids)

    def _bench_embed(self, obs: dict[str, torch.Tensor], prefix: str, batch_shape: torch.Size, *, role: int) -> torch.Tensor:
        card_ids = _require_batch_prefix(obs, f"{prefix}_card_ids", batch_shape).to(dtype=torch.long)
        hp = _require_batch_prefix(obs, f"{prefix}_hp", batch_shape).to(dtype=torch.float32)
        energy_counts = _require_batch_prefix(obs, f"{prefix}_energy_counts", batch_shape).to(dtype=torch.float32)
        tool_counts = _require_batch_prefix(obs, f"{prefix}_tool_counts", batch_shape).to(dtype=torch.float32)
        if card_ids.shape != hp.shape or hp.shape != energy_counts.shape or hp.shape != tool_counts.shape:
            raise ValueError(f"{prefix} card, hp, energy, and tool tensors must have matching shapes.")
        slot = torch.cat(
            (
                self._card_embed(card_ids, role=role),
                (hp / 100.0).unsqueeze(-1),
                (energy_counts / 10.0).unsqueeze(-1),
                (tool_counts / 4.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded_slots = self.bench_trunk(slot)
        mask = card_ids.ge(0).unsqueeze(-1)
        totals = (encoded_slots * mask).sum(dim=-2)
        counts = mask.sum(dim=-2).clamp_min(1)
        return totals / counts

    def _stadium_card_embed(self, obs: dict[str, torch.Tensor], batch_shape: torch.Size) -> torch.Tensor:
        ids = _require_batch_prefix(obs, "stadium_card_id", batch_shape).to(dtype=torch.long)
        if ids.shape != (*batch_shape, 1):
            raise ValueError("obs['stadium_card_id'] must have shape [..., 1].")
        return self._card_embed(ids.squeeze(-1), role=CARD_ROLE_STADIUM)

    def _option_embed(self, option_features: torch.Tensor) -> torch.Tensor:
        features = option_features.to(dtype=torch.long)
        option_type = self.enum_embedding(_offset_ids(features[..., 0]))
        area = self.enum_embedding(_offset_ids(features[..., 2]))
        player_index = self.enum_embedding(_offset_ids(features[..., 4]))
        in_play_area = self.enum_embedding(_offset_ids(features[..., 8]))
        in_play_index = self.enum_embedding(_offset_ids(features[..., 9]))
        attack = self._attack_embed(features[..., 10])
        card = self._card_embed(features[..., 11], role=CARD_ROLE_OPTION)
        special_condition = self.enum_embedding(_offset_ids(features[..., 13]))
        numeric_indices = [index for index in range(self.option_feature_size) if index not in {0, 2, 4, 8, 9, 10, 11, 13}]
        numeric = option_features[..., numeric_indices].to(dtype=torch.float32) / 100.0
        return self.option_projection(
            torch.cat(
                (
                    option_type,
                    area,
                    player_index,
                    in_play_area,
                    in_play_index,
                    attack,
                    card,
                    special_condition,
                    numeric,
                ),
                dim=-1,
            )
        )

    def _card_embed(self, ids: torch.Tensor, *, role: int = 0) -> torch.Tensor:
        learned = self.card_embedding(_offset_ids(ids, self.card_embedding.num_embeddings - 1))
        metadata = _lookup_metadata(self.card_features, ids).to(dtype=torch.float32)
        metadata = _normalize_card_features(metadata)
        mask = ids.ge(0).unsqueeze(-1)
        role_ids = torch.full_like(ids, role).clamp_min(0).clamp_max(CARD_ROLE_COUNT - 1)
        return learned + (self.card_metadata_projection(metadata) + self.card_role_embedding(role_ids)) * mask

    def _attack_embed(self, ids: torch.Tensor) -> torch.Tensor:
        learned = self.attack_embedding(_offset_ids(ids, self.attack_embedding.num_embeddings - 1))
        features = _lookup_metadata(self.attack_features, ids).to(dtype=torch.float32)
        cost_counts = _lookup_metadata(self.attack_cost_counts, ids).to(dtype=torch.float32).unsqueeze(-1)
        features = _normalize_attack_features(torch.cat((features, cost_counts), dim=-1))
        cost_energy = _lookup_metadata(self.attack_cost_energy_types, ids).to(dtype=torch.long)
        cost_embed = _masked_mean_embedding(self.energy_embedding, cost_energy)
        mask = ids.ge(0).unsqueeze(-1)
        metadata = self.attack_metadata_projection(torch.cat((features, cost_embed), dim=-1))
        return learned + metadata * mask

    def _masked_mean_card_embedding(self, ids: torch.Tensor, *, role: int) -> torch.Tensor:
        embedded = self._card_embed(ids, role=role)
        mask = ids.ge(0).unsqueeze(-1)
        totals = (embedded * mask).sum(dim=-2)
        counts = mask.sum(dim=-2).clamp_min(1)
        return totals / counts


def _offset_embedding(vocab_size: int, embedding_dim: int) -> torch.nn.Embedding:
    if vocab_size <= 0:
        raise ValueError("embedding vocab sizes must be positive.")
    return torch.nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)


def _metadata_buffer(values: torch.Tensor | None, rows: int, cols: int, *, fill: int = 0) -> torch.Tensor:
    out = torch.full((rows, cols), fill, dtype=torch.long)
    if values is None:
        return out
    source = torch.as_tensor(values, dtype=torch.long)
    if source.ndim != 2 or source.shape[1] != cols:
        raise ValueError(f"metadata table must have shape [N, {cols}].")
    copy_rows = min(rows, source.shape[0])
    out[:copy_rows] = source[:copy_rows]
    return out


def _metadata_vector_buffer(values: torch.Tensor | None, rows: int, *, fill: int = 0) -> torch.Tensor:
    out = torch.full((rows,), fill, dtype=torch.long)
    if values is None:
        return out
    source = torch.as_tensor(values, dtype=torch.long)
    if source.ndim != 1:
        raise ValueError("metadata vector must have shape [N].")
    copy_rows = min(rows, source.shape[0])
    out[:copy_rows] = source[:copy_rows]
    return out


def _lookup_metadata(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    indices = ids.clamp_min(0).clamp_max(table.shape[0] - 1).to(dtype=torch.long)
    values = table[indices]
    return values * ids.ge(0).to(dtype=values.dtype).unsqueeze(-1) if values.ndim > ids.ndim else values * ids.ge(0).to(dtype=values.dtype)


def _normalize_card_features(features: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(
        [1, 16, 340, 5, 16, 16, 16, 1, 1, 1, 1, 1, 1, 1],
        dtype=features.dtype,
        device=features.device,
    )
    normalized = features / scale
    normalized[..., 4:6] = (features[..., 4:6] + 1.0) / 17.0
    return normalized


def _normalize_attack_features(features: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor([1, 400, 8, 1, 64, 512, 8], dtype=features.dtype, device=features.device)
    return features / scale


def _offset_ids(ids: torch.Tensor, vocab_size: int | None = None) -> torch.Tensor:
    offset = ids.clamp_min(-1) + 1
    if vocab_size is not None:
        offset = offset.clamp_max(vocab_size)
    return offset.to(dtype=torch.long)


def _masked_mean_embedding(embedding: torch.nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    embedded = embedding(_offset_ids(ids, embedding.num_embeddings - 1))
    mask = ids.ge(0).unsqueeze(-1)
    totals = (embedded * mask).sum(dim=-2)
    counts = mask.sum(dim=-2).clamp_min(1)
    return totals / counts


def _require_last_dim(obs: dict[str, torch.Tensor], key: str, size: int) -> torch.Tensor:
    if key not in obs:
        raise ValueError(f"obs is missing required tensor {key!r}.")
    value = obs[key]
    if value.shape[-1:] != (size,):
        raise ValueError(f"obs[{key!r}] must have last dimension {size}.")
    return value


def _require_batch_prefix(obs: dict[str, torch.Tensor], key: str, batch_shape: torch.Size) -> torch.Tensor:
    if key not in obs:
        raise ValueError(f"obs is missing required tensor {key!r}.")
    value = obs[key]
    if value.shape[: len(batch_shape)] != batch_shape:
        raise ValueError(f"obs[{key!r}] batch shape must match scalar/zones batch shape.")
    return value


__all__ = ["ObservationEncoder"]

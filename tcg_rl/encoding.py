"""Fixed-shape observation encoders for RL policies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from dataclasses import fields
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pokemon_env.card_db import CardDB


ENCODER_SCHEMA_VERSION = 2
DEFAULT_OBSERVATION_NORMALIZATION_KEYS = (
    "scalar",
    "zones",
    "bench_hp",
    "bench_energy_counts",
    "bench_tool_counts",
    "opponent_bench_hp",
    "opponent_bench_energy_counts",
    "opponent_bench_tool_counts",
)


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    max_options: int = 128
    max_hand: int = 32
    max_bench: int = 5
    max_attached_energy: int = 16
    max_attached_tools: int = 4
    option_feature_size: int = 14


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class TensorSpec:
    observation: dict[str, TensorFieldSpec]
    categorical_action: TensorFieldSpec
    padded_action: dict[str, TensorFieldSpec]


@dataclass(frozen=True, slots=True)
class EncoderSchema:
    version: int
    config: dict[str, int]
    observation: dict[str, TensorFieldSpec]
    categorical_action: TensorFieldSpec
    padded_action: dict[str, TensorFieldSpec]
    fingerprint: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "config": dict(self.config),
            "observation": _field_spec_mapping_to_dict(self.observation),
            "categorical_action": _field_spec_to_dict(self.categorical_action),
            "padded_action": _field_spec_mapping_to_dict(self.padded_action),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class TensorObservationStats:
    """Running per-feature moments for encoded tensor observations."""

    count: int
    mean: dict[str, torch.Tensor]
    m2: dict[str, torch.Tensor]

    @property
    def variance(self) -> dict[str, torch.Tensor]:
        torch = _torch()
        if self.count <= 1:
            return {key: torch.zeros_like(value) for key, value in self.mean.items()}
        return {key: value / self.count for key, value in self.m2.items()}

    @property
    def std(self) -> dict[str, torch.Tensor]:
        return {key: value.clamp_min(0.0).sqrt() for key, value in self.variance.items()}

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": {key: value.clone() for key, value in self.mean.items()},
            "m2": {key: value.clone() for key, value in self.m2.items()},
        }


def encoder_schema(config: EncoderConfig | None = None) -> EncoderSchema:
    """Return a stable schema descriptor for checkpoint compatibility checks."""

    config = config or EncoderConfig()
    spec = tensor_spec(config)
    payload = {
        "version": ENCODER_SCHEMA_VERSION,
        "config": _encoder_config_to_dict(config),
        "observation": _field_spec_mapping_to_dict(spec.observation),
        "categorical_action": _field_spec_to_dict(spec.categorical_action),
        "padded_action": _field_spec_mapping_to_dict(spec.padded_action),
    }
    return EncoderSchema(
        version=ENCODER_SCHEMA_VERSION,
        config=payload["config"],
        observation=spec.observation,
        categorical_action=spec.categorical_action,
        padded_action=spec.padded_action,
        fingerprint=_schema_fingerprint(payload),
    )


def validate_encoder_schema(saved_schema: EncoderSchema | dict, config: EncoderConfig | None = None) -> None:
    """Raise if a saved encoder schema is incompatible with ``config``."""

    current = encoder_schema(config)
    saved = saved_schema.to_dict() if isinstance(saved_schema, EncoderSchema) else dict(saved_schema)
    if saved.get("version") != current.version:
        raise ValueError(f"Encoder schema version mismatch: saved={saved.get('version')!r}, current={current.version!r}.")
    if saved.get("fingerprint") != current.fingerprint:
        raise ValueError(
            "Encoder schema fingerprint mismatch: "
            f"saved={saved.get('fingerprint')!r}, current={current.fingerprint!r}."
        )


def encoded_observation_spec(config: EncoderConfig | None = None) -> dict[str, TensorFieldSpec]:
    config = config or EncoderConfig()
    return {
        "scalar": TensorFieldSpec((14,), "long"),
        "zones": TensorFieldSpec((28,), "long"),
        "hand_card_ids": TensorFieldSpec((config.max_hand,), "long"),
        "bench_card_ids": TensorFieldSpec((config.max_bench,), "long"),
        "bench_hp": TensorFieldSpec((config.max_bench,), "long"),
        "bench_energy_counts": TensorFieldSpec((config.max_bench,), "long"),
        "bench_tool_counts": TensorFieldSpec((config.max_bench,), "long"),
        "opponent_bench_card_ids": TensorFieldSpec((config.max_bench,), "long"),
        "opponent_bench_hp": TensorFieldSpec((config.max_bench,), "long"),
        "opponent_bench_energy_counts": TensorFieldSpec((config.max_bench,), "long"),
        "opponent_bench_tool_counts": TensorFieldSpec((config.max_bench,), "long"),
        "active_energy_types": TensorFieldSpec((2, config.max_attached_energy), "long"),
        "active_tool_card_ids": TensorFieldSpec((2, config.max_attached_tools), "long"),
        "stadium_card_id": TensorFieldSpec((1,), "long"),
        "option_features": TensorFieldSpec((config.max_options, config.option_feature_size), "long"),
        "action_mask": TensorFieldSpec((config.max_options,), "bool"),
        "categorical_action_mask": TensorFieldSpec((config.max_options + 1,), "bool"),
        "selection_count_mask": TensorFieldSpec((config.max_options + 1,), "bool"),
    }


def tensor_spec(config: EncoderConfig | None = None) -> TensorSpec:
    config = config or EncoderConfig()
    return TensorSpec(
        observation=encoded_observation_spec(config),
        categorical_action=TensorFieldSpec((), "long"),
        padded_action={
            "padded_indices": TensorFieldSpec((config.max_options,), "long"),
            "counts": TensorFieldSpec((), "long"),
        },
    )


def tensor_dtype_for_encoded_key(key: str, torch):
    if key.endswith("_mask") or key == "action_mask":
        return torch.bool
    return torch.long


def numpy_dtype_for_encoded_key(key: str):
    if key.endswith("_mask") or key == "action_mask":
        return np.bool_
    return np.int64


def encoded_arrays_to_numpy(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.ascontiguousarray(np.asarray(value, dtype=numpy_dtype_for_encoded_key(key)))
        for key, value in arrays.items()
    }


def encoded_arrays_to_tensors(arrays: dict[str, Any], device=None) -> dict:
    torch = _torch()
    device = _tensor_device(device)
    numpy_arrays = encoded_arrays_to_numpy(arrays)
    tensors = {key: torch.from_numpy(value) for key, value in numpy_arrays.items()}
    if device.type == "cpu":
        return tensors
    return {key: value.to(device=device) for key, value in tensors.items()}


def update_tensor_observation_stats(
    stats: TensorObservationStats | None,
    tensor_obs: dict[str, Any],
    *,
    config: EncoderConfig | None = None,
    keys: tuple[str, ...] | list[str] | None = None,
) -> TensorObservationStats:
    """Update running per-feature moments from a tensor observation batch."""

    torch = _torch()
    config = config or EncoderConfig()
    keys = tuple(keys or DEFAULT_OBSERVATION_NORMALIZATION_KEYS)
    specs = encoded_observation_spec(config)
    count = 0 if stats is None else int(stats.count)
    means = {} if stats is None else {key: value.clone() for key, value in stats.mean.items()}
    m2 = {} if stats is None else {key: value.clone() for key, value in stats.m2.items()}

    for key in keys:
        if key not in tensor_obs:
            raise ValueError(f"tensor_obs is missing normalization key: {key}")
        if key not in specs:
            raise ValueError(f"Unknown encoded observation key: {key}")
        values = _observation_samples(tensor_obs[key], specs[key], key)
        batch_count = int(values.shape[0])
        if batch_count == 0:
            continue
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        if key not in means:
            if count:
                raise ValueError(f"Existing stats are missing normalization key: {key}")
            means[key] = batch_mean
            m2[key] = batch_m2
            continue
        if means[key].shape != batch_mean.shape:
            raise ValueError(f"Existing stats shape for {key!r} does not match observation shape.")
        total = count + batch_count
        delta = batch_mean - means[key]
        means[key] = means[key] + delta * (batch_count / total)
        m2[key] = m2[key] + batch_m2 + delta.pow(2) * count * batch_count / total
    sample_count = _observation_sample_count(tensor_obs, specs, keys)
    return TensorObservationStats(count=count + sample_count, mean=means, m2=m2)


def normalize_tensor_observation(
    tensor_obs: dict[str, Any],
    stats: TensorObservationStats,
    *,
    keys: tuple[str, ...] | list[str] | None = None,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Return a copy with selected observation tensors normalized as float32."""

    torch = _torch()
    if eps <= 0:
        raise ValueError("eps must be positive.")
    keys = tuple(keys or stats.mean.keys())
    normalized = dict(tensor_obs)
    std = stats.std
    for key in keys:
        if key not in tensor_obs:
            raise ValueError(f"tensor_obs is missing normalization key: {key}")
        if key not in stats.mean:
            raise ValueError(f"stats are missing normalization key: {key}")
        value = torch.as_tensor(tensor_obs[key], dtype=torch.float32)
        normalized[key] = (value - stats.mean[key].to(device=value.device)) / std[key].to(device=value.device).clamp_min(eps)
    return normalized


def encode_card_db_metadata(
    card_db: CardDB,
    *,
    max_card_id: int | None = None,
    max_attack_id: int | None = None,
    max_attacks_per_card: int = 4,
    max_attack_cost: int = 8,
    max_evolution_links: int = 8,
) -> dict[str, list]:
    """Encode immutable card and attack metadata for policy feature tables."""

    if max_attacks_per_card <= 0:
        raise ValueError("max_attacks_per_card must be positive.")
    if max_attack_cost <= 0:
        raise ValueError("max_attack_cost must be positive.")
    if max_evolution_links <= 0:
        raise ValueError("max_evolution_links must be positive.")
    max_card_id = _metadata_table_size(max_card_id, card_db.cards)
    max_attack_id = _metadata_table_size(max_attack_id, card_db.attacks)
    card_features = [[0] * 14 for _ in range(max_card_id + 1)]
    card_raw_features = [[0] * 12 for _ in range(max_card_id + 1)]
    card_attack_ids = [[-1] * max_attacks_per_card for _ in range(max_card_id + 1)]
    card_evolves_from_ids = [[-1] * max_evolution_links for _ in range(max_card_id + 1)]
    card_evolves_to_ids = [[-1] * max_evolution_links for _ in range(max_card_id + 1)]
    attack_features = [[0] * 6 for _ in range(max_attack_id + 1)]
    attack_cost_energy_types = [[-1] * max_attack_cost for _ in range(max_attack_id + 1)]
    attack_cost_counts = [0 for _ in range(max_attack_id + 1)]
    card_ids_by_name = _card_ids_by_name(card_db)
    evolves_to_ids = _evolves_to_ids(card_db, card_ids_by_name)

    for card_id, card in card_db.cards.items():
        if card_id > max_card_id:
            continue
        card_features[card_id] = [
            1,
            int(card.cardType),
            card.hp,
            card.retreatCost,
            -1 if card.weakness is None else int(card.weakness),
            -1 if card.resistance is None else int(card.resistance),
            int(card.energyType),
            int(card.basic),
            int(card.stage1),
            int(card.stage2),
            int(card.ex),
            int(card.megaEx),
            int(card.tera),
            int(card.aceSpec),
        ]
        raw_effect_text = _raw_text(card.raw, "effect explanation", "effect", "text", "attack text", "skill text", "ability text")
        card_raw_features[card_id] = [
            1,
            _stable_text_id(_raw_text(card.raw, "expansion", "set", "set code")),
            _parse_metadata_int(_raw_text(card.raw, "collection no.", "collection no", "collection number", "number")),
            _stable_text_id(_raw_text(card.raw, "rule")),
            _stable_text_id(_raw_text(card.raw, "category")),
            _stable_text_id(_raw_text(card.raw, "rarity")),
            _stable_text_id(_raw_text(card.raw, "regulation mark", "regulation")),
            len(card.name),
            0 if card.evolvesFrom is None else len(card.evolvesFrom),
            int(bool(raw_effect_text)),
            len(raw_effect_text),
            len(card.skills),
        ]
        card_attack_ids[card_id] = _pad([attack_id for attack_id in card.attacks], max_attacks_per_card, fill=-1)
        if card.evolvesFrom:
            card_evolves_from_ids[card_id] = _pad(
                card_ids_by_name.get(_metadata_name_key(card.evolvesFrom), ()),
                max_evolution_links,
                fill=-1,
            )
        card_evolves_to_ids[card_id] = _pad(evolves_to_ids.get(card_id, ()), max_evolution_links, fill=-1)

    for attack_id, attack in card_db.attacks.items():
        if attack_id > max_attack_id:
            continue
        cost = [int(energy) for energy in attack.energies]
        attack_features[attack_id] = [
            1,
            attack.damage,
            len(cost),
            int(bool(attack.text)),
            len(attack.name),
            len(attack.text),
        ]
        attack_cost_energy_types[attack_id] = _pad(cost, max_attack_cost, fill=-1)
        attack_cost_counts[attack_id] = min(len(cost), max_attack_cost)

    return {
        "card_features": card_features,
        "card_raw_features": card_raw_features,
        "card_attack_ids": card_attack_ids,
        "card_evolves_from_ids": card_evolves_from_ids,
        "card_evolves_to_ids": card_evolves_to_ids,
        "attack_features": attack_features,
        "attack_cost_energy_types": attack_cost_energy_types,
        "attack_cost_counts": attack_cost_counts,
    }


def encode_card_db_metadata_to_tensors(card_db: CardDB, device=None, **kwargs) -> dict:
    torch = _torch()
    device = _tensor_device(device)
    arrays = encode_card_db_metadata(card_db, **kwargs)
    return {key: torch.as_tensor(value, dtype=torch.long, device=device) for key, value in arrays.items()}


def encode_observation(obs: dict[str, Any], config: EncoderConfig | None = None) -> dict[str, list[int] | list[list[int]]]:
    config = config or EncoderConfig()
    current = obs["current"]
    select = obs["select"]
    players = current["players"]
    active0 = _active(players[0])
    active1 = _active(players[1])
    your = current["yourIndex"]
    visible_hand = players[your]["hand"] or []

    scalar = [
        current["turn"],
        current["turnActionCount"],
        your,
        current["firstPlayer"],
        current["result"],
        int(current["supporterPlayed"]),
        int(current["stadiumPlayed"]),
        int(current["energyAttached"]),
        int(current["retreated"]),
        -1 if select is None else select["type"],
        -1 if select is None else select["context"],
        0 if select is None else select["minCount"],
        0 if select is None else select["maxCount"],
        0 if select is None else len(select["option"]),
    ]
    zones = [
        players[0]["deckCount"],
        players[0]["handCount"],
        len(players[0]["prize"]),
        len(players[0]["discard"]),
        len(players[0]["bench"]),
        -1 if active0 is None else active0["id"],
        0 if active0 is None else active0["hp"],
        0 if active0 is None else active0["maxHp"],
        len(active0["energies"]) if active0 is not None else 0,
        int(players[0]["poisoned"]),
        int(players[0]["burned"]),
        int(players[0]["asleep"]),
        int(players[0]["paralyzed"]),
        int(players[0]["confused"]),
        players[1]["deckCount"],
        players[1]["handCount"],
        len(players[1]["prize"]),
        len(players[1]["discard"]),
        len(players[1]["bench"]),
        -1 if active1 is None else active1["id"],
        0 if active1 is None else active1["hp"],
        0 if active1 is None else active1["maxHp"],
        len(active1["energies"]) if active1 is not None else 0,
        int(players[1]["poisoned"]),
        int(players[1]["burned"]),
        int(players[1]["asleep"]),
        int(players[1]["paralyzed"]),
        int(players[1]["confused"]),
    ]
    return {
        "scalar": scalar,
        "zones": zones,
        "hand_card_ids": _pad([card["id"] for card in visible_hand], config.max_hand, fill=-1),
        "bench_card_ids": _pad(_bench_ids(players[your], config.max_bench), config.max_bench, fill=-1),
        "bench_hp": _pad(_bench_hp(players[your], config.max_bench), config.max_bench, fill=0),
        "bench_energy_counts": _pad(_bench_energy_counts(players[your], config.max_bench), config.max_bench, fill=0),
        "bench_tool_counts": _pad(_bench_tool_counts(players[your], config.max_bench), config.max_bench, fill=0),
        "opponent_bench_card_ids": _pad(_bench_ids(players[1 - your], config.max_bench), config.max_bench, fill=-1),
        "opponent_bench_hp": _pad(_bench_hp(players[1 - your], config.max_bench), config.max_bench, fill=0),
        "opponent_bench_energy_counts": _pad(_bench_energy_counts(players[1 - your], config.max_bench), config.max_bench, fill=0),
        "opponent_bench_tool_counts": _pad(_bench_tool_counts(players[1 - your], config.max_bench), config.max_bench, fill=0),
        "active_energy_types": _active_energy_types(players, config.max_attached_energy),
        "active_tool_card_ids": _active_tool_card_ids(players, config.max_attached_tools),
        "stadium_card_id": [_stadium_card_id(current)],
        "option_features": encode_options(select, config),
        "action_mask": legal_action_mask(select, config.max_options),
        "categorical_action_mask": categorical_action_mask(select, config.max_options + 1),
        "selection_count_mask": selection_count_mask(select, config.max_options + 1),
    }


def encode_observation_numpy(obs: dict[str, Any], config: EncoderConfig | None = None) -> dict[str, np.ndarray]:
    """Encode an observation as contiguous fixed-shape NumPy leaves."""

    return encoded_arrays_to_numpy(encode_observation(obs, config))


def encode_observation_into(
    obs: dict[str, Any],
    out: dict[str, np.ndarray],
    config: EncoderConfig | None = None,
    *,
    debug_equivalence: bool = False,
    validate_output: bool = True,
) -> None:
    """Encode an observation into preallocated NumPy leaves."""

    config = config or EncoderConfig()
    if validate_output:
        _assert_output_spec(out, config)
    encoded = encode_observation(obs, config)
    for key, value in encoded.items():
        out[key][...] = np.asarray(value, dtype=numpy_dtype_for_encoded_key(key))
    if debug_equivalence:
        _assert_encoded_equal(out, encode_observation_numpy(obs, config))


def encode_state(state: Any, config: EncoderConfig | None = None) -> dict[str, list[int] | list[list[int]]]:
    """Encode internal engine state directly for RL hot paths."""

    config = config or EncoderConfig()
    players = state.players
    active0 = players[0].active
    active1 = players[1].active
    your = state.yourIndex
    visible_hand = players[your].hand
    select = state.pending

    scalar = [
        state.turn,
        state.turnActionCount,
        your,
        state.firstPlayer,
        state.result,
        int(state.supporterPlayed),
        int(state.stadiumPlayed),
        int(state.energyAttached),
        int(state.retreated),
        -1 if select is None else int(select.type),
        -1 if select is None else int(select.context),
        0 if select is None else select.minCount,
        0 if select is None else select.maxCount,
        0 if select is None else len(select.option),
    ]
    zones = [
        len(players[0].deck),
        len(players[0].hand),
        len(players[0].prize),
        len(players[0].discard),
        len(players[0].bench),
        -1 if active0 is None else active0.id,
        0 if active0 is None else active0.hp,
        0 if active0 is None else active0.maxHp,
        len(active0.energies) if active0 is not None else 0,
        int(players[0].poisoned),
        int(players[0].burned),
        int(players[0].asleep),
        int(players[0].paralyzed),
        int(players[0].confused),
        len(players[1].deck),
        len(players[1].hand),
        len(players[1].prize),
        len(players[1].discard),
        len(players[1].bench),
        -1 if active1 is None else active1.id,
        0 if active1 is None else active1.hp,
        0 if active1 is None else active1.maxHp,
        len(active1.energies) if active1 is not None else 0,
        int(players[1].poisoned),
        int(players[1].burned),
        int(players[1].asleep),
        int(players[1].paralyzed),
        int(players[1].confused),
    ]
    return {
        "scalar": scalar,
        "zones": zones,
        "hand_card_ids": _pad([card.id for card in visible_hand], config.max_hand, fill=-1),
        "bench_card_ids": _pad(_bench_ids_from_state(players[your], config.max_bench), config.max_bench, fill=-1),
        "bench_hp": _pad(_bench_hp_from_state(players[your], config.max_bench), config.max_bench, fill=0),
        "bench_energy_counts": _pad(_bench_energy_counts_from_state(players[your], config.max_bench), config.max_bench, fill=0),
        "bench_tool_counts": _pad(_bench_tool_counts_from_state(players[your], config.max_bench), config.max_bench, fill=0),
        "opponent_bench_card_ids": _pad(_bench_ids_from_state(players[1 - your], config.max_bench), config.max_bench, fill=-1),
        "opponent_bench_hp": _pad(_bench_hp_from_state(players[1 - your], config.max_bench), config.max_bench, fill=0),
        "opponent_bench_energy_counts": _pad(
            _bench_energy_counts_from_state(players[1 - your], config.max_bench),
            config.max_bench,
            fill=0,
        ),
        "opponent_bench_tool_counts": _pad(
            _bench_tool_counts_from_state(players[1 - your], config.max_bench),
            config.max_bench,
            fill=0,
        ),
        "active_energy_types": _active_energy_types_from_state(players, config.max_attached_energy),
        "active_tool_card_ids": _active_tool_card_ids_from_state(players, config.max_attached_tools),
        "stadium_card_id": [_stadium_card_id_from_state(state)],
        "option_features": encode_select_options(select, config),
        "action_mask": legal_action_mask_from_select(select, config.max_options),
        "categorical_action_mask": categorical_action_mask_from_select(select, config.max_options + 1),
        "selection_count_mask": selection_count_mask_from_select(select, config.max_options + 1),
    }


def encode_state_numpy(state: Any, config: EncoderConfig | None = None) -> dict[str, np.ndarray]:
    """Encode internal engine state as contiguous fixed-shape NumPy leaves."""

    return encoded_arrays_to_numpy(encode_state(state, config))


def encode_state_into(
    state: Any,
    out: dict[str, np.ndarray],
    config: EncoderConfig | None = None,
    *,
    debug_equivalence: bool = False,
    validate_output: bool = True,
) -> None:
    """Encode internal engine state into preallocated NumPy leaves.

    Every fixed-shape field is fully overwritten, including padded regions, so
    reused shared-memory buffers cannot retain values from previous prompts.
    """

    config = config or EncoderConfig()
    if validate_output:
        _assert_output_spec(out, config)
    players = state.players
    active0 = players[0].active
    active1 = players[1].active
    your = state.yourIndex
    select = state.pending

    out["scalar"][...] = (
        state.turn,
        state.turnActionCount,
        your,
        state.firstPlayer,
        state.result,
        int(state.supporterPlayed),
        int(state.stadiumPlayed),
        int(state.energyAttached),
        int(state.retreated),
        -1 if select is None else int(select.type),
        -1 if select is None else int(select.context),
        0 if select is None else select.minCount,
        0 if select is None else select.maxCount,
        0 if select is None else len(select.option),
    )
    out["zones"][...] = (
        len(players[0].deck),
        len(players[0].hand),
        len(players[0].prize),
        len(players[0].discard),
        len(players[0].bench),
        -1 if active0 is None else active0.id,
        0 if active0 is None else active0.hp,
        0 if active0 is None else active0.maxHp,
        len(active0.energies) if active0 is not None else 0,
        int(players[0].poisoned),
        int(players[0].burned),
        int(players[0].asleep),
        int(players[0].paralyzed),
        int(players[0].confused),
        len(players[1].deck),
        len(players[1].hand),
        len(players[1].prize),
        len(players[1].discard),
        len(players[1].bench),
        -1 if active1 is None else active1.id,
        0 if active1 is None else active1.hp,
        0 if active1 is None else active1.maxHp,
        len(active1.energies) if active1 is not None else 0,
        int(players[1].poisoned),
        int(players[1].burned),
        int(players[1].asleep),
        int(players[1].paralyzed),
        int(players[1].confused),
    )

    _write_card_ids(out["hand_card_ids"], (card.id for card in players[your].hand), fill=-1)
    _write_pokemon_attr(out["bench_card_ids"], players[your].bench, "id", fill=-1)
    _write_pokemon_attr(out["bench_hp"], players[your].bench, "hp", fill=0)
    _write_pokemon_len(out["bench_energy_counts"], players[your].bench, "energies")
    _write_pokemon_len(out["bench_tool_counts"], players[your].bench, "tools")
    _write_pokemon_attr(out["opponent_bench_card_ids"], players[1 - your].bench, "id", fill=-1)
    _write_pokemon_attr(out["opponent_bench_hp"], players[1 - your].bench, "hp", fill=0)
    _write_pokemon_len(out["opponent_bench_energy_counts"], players[1 - your].bench, "energies")
    _write_pokemon_len(out["opponent_bench_tool_counts"], players[1 - your].bench, "tools")
    _write_active_energy_types(out["active_energy_types"], players)
    _write_active_tool_card_ids(out["active_tool_card_ids"], players)
    out["stadium_card_id"][...] = (_stadium_card_id_from_state(state),)
    _write_select_options(out["option_features"], select)
    _write_legal_action_mask(out["action_mask"], select)
    _write_categorical_action_mask(out["categorical_action_mask"], select)
    _write_selection_count_mask(out["selection_count_mask"], select)

    if debug_equivalence:
        _assert_encoded_equal(out, encode_state_numpy(state, config))


def encode_options(select: dict[str, Any] | None, config: EncoderConfig | None = None) -> list[list[int]]:
    config = config or EncoderConfig()
    rows: list[list[int]] = []
    options = [] if select is None else select["option"]
    for option in options[: config.max_options]:
        rows.append(
            [
                _none_to(option.get("type"), -1),
                _none_to(option.get("number"), -1),
                _none_to(option.get("area"), -1),
                _none_to(option.get("index"), -1),
                _none_to(option.get("playerIndex"), -1),
                _none_to(option.get("toolIndex"), -1),
                _none_to(option.get("energyIndex"), -1),
                _none_to(option.get("count"), -1),
                _none_to(option.get("inPlayArea"), -1),
                _none_to(option.get("inPlayIndex"), -1),
                _none_to(option.get("attackId"), -1),
                _none_to(option.get("cardId"), -1),
                _none_to(option.get("serial"), -1),
                _none_to(option.get("specialConditionType"), -1),
            ]
        )
    while len(rows) < config.max_options:
        rows.append([-1] * config.option_feature_size)
    return rows


def encode_select_options(select: Any | None, config: EncoderConfig | None = None) -> list[list[int]]:
    config = config or EncoderConfig()
    rows: list[list[int]] = []
    options = [] if select is None else select.option
    for option in options[: config.max_options]:
        rows.append(
            [
                _none_to(option.type, -1),
                _none_to(option.number, -1),
                _none_to(option.area, -1),
                _none_to(option.index, -1),
                _none_to(option.playerIndex, -1),
                _none_to(option.toolIndex, -1),
                _none_to(option.energyIndex, -1),
                _none_to(option.count, -1),
                _none_to(option.inPlayArea, -1),
                _none_to(option.inPlayIndex, -1),
                _none_to(option.attackId, -1),
                _none_to(option.cardId, -1),
                _none_to(option.serial, -1),
                _none_to(option.specialConditionType, -1),
            ]
        )
    while len(rows) < config.max_options:
        rows.append([-1] * config.option_feature_size)
    return rows


def legal_action_mask(select: dict[str, Any] | None, max_options: int = 128) -> list[int]:
    count = 0 if select is None else min(len(select["option"]), max_options)
    return [1 if i < count else 0 for i in range(max_options)]


def legal_action_mask_from_select(select: Any | None, max_options: int = 128) -> list[int]:
    count = 0 if select is None else min(len(select.option), max_options)
    return [1 if i < count else 0 for i in range(max_options)]


def categorical_action_mask(select: dict[str, Any] | None, size: int = 129) -> list[int]:
    if select is None:
        return [0] * size
    mask = [0] * size
    mask[0] = 1 if select["minCount"] == 0 else 0
    if select["minCount"] <= 1 <= select["maxCount"]:
        option_slots = min(len(select["option"]), size - 1)
        for index in range(option_slots):
            mask[index + 1] = 1
    return mask


def categorical_action_mask_from_select(select: Any | None, size: int = 129) -> list[int]:
    if select is None:
        return [0] * size
    mask = [0] * size
    mask[0] = 1 if select.minCount == 0 else 0
    if select.minCount <= 1 <= select.maxCount:
        option_slots = min(len(select.option), size - 1)
        for index in range(option_slots):
            mask[index + 1] = 1
    return mask


def selection_count_mask(select: dict[str, Any] | None, size: int = 129) -> list[int]:
    if select is None:
        return [0] * size
    mask = [0] * size
    legal_options = len(select["option"])
    min_count = int(select["minCount"])
    max_count = min(int(select["maxCount"]), legal_options, size - 1)
    for count in range(min_count, max_count + 1):
        mask[count] = 1
    return mask


def selection_count_mask_from_select(select: Any | None, size: int = 129) -> list[int]:
    if select is None:
        return [0] * size
    mask = [0] * size
    legal_options = len(select.option)
    min_count = int(select.minCount)
    max_count = min(int(select.maxCount), legal_options, size - 1)
    for count in range(min_count, max_count + 1):
        mask[count] = 1
    return mask


def option_indices_to_actions(indices: list[int] | tuple[int, ...], masks: list[list[int]]) -> list[list[int]]:
    """Convert one selected padded option index per env into CABT-style actions."""

    if len(indices) != len(masks):
        raise ValueError("indices and masks must have the same length.")
    _require_integral_values(indices, "indices")
    actions: list[list[int]] = []
    for index, mask in zip(indices, masks, strict=True):
        if index < 0 or index >= len(mask) or not mask[index]:
            raise ValueError(f"Selected illegal option index: {index}")
        actions.append([int(index)])
    return actions


def categorical_indices_to_actions(indices: list[int] | tuple[int, ...], masks: list[list[int]]) -> list[list[int]]:
    """Convert fixed categorical action indices into CABT-style actions.

    Slot 0 maps to ``[]`` when the current prompt allows zero selections.
    Slots 1..N map to selecting option indices 0..N-1.
    """

    if len(indices) != len(masks):
        raise ValueError("indices and masks must have the same length.")
    _require_integral_values(indices, "indices")
    actions: list[list[int]] = []
    for index, mask in zip(indices, masks, strict=True):
        if index < 0 or index >= len(mask) or not mask[index]:
            raise ValueError(f"Selected illegal categorical action index: {index}")
        actions.append([] if index == 0 else [int(index) - 1])
    return actions


def padded_indices_to_actions(
    padded_indices: list[list[int]] | tuple[tuple[int, ...], ...],
    counts: list[int] | tuple[int, ...],
    masks: list[list[int]],
    min_counts: list[int] | tuple[int, ...] | None = None,
    max_counts: list[int] | tuple[int, ...] | None = None,
) -> list[list[int]]:
    """Convert padded per-env option indices into CABT-style selections."""

    if len(padded_indices) != len(counts) or len(counts) != len(masks):
        raise ValueError("padded_indices, counts, and masks must have the same batch length.")
    if min_counts is not None and len(min_counts) != len(counts):
        raise ValueError("min_counts length must match counts length.")
    if max_counts is not None and len(max_counts) != len(counts):
        raise ValueError("max_counts length must match counts length.")
    _require_integral_values(counts, "counts")
    actions: list[list[int]] = []
    if min_counts is None:
        min_counts = [0 for _ in counts]
    if max_counts is None:
        max_counts = [len(row) for row in padded_indices]
    for row, count, mask, min_count, max_count in zip(padded_indices, counts, masks, min_counts, max_counts, strict=True):
        _require_integral_values(row[:count], "padded_indices")
        if count < 0 or count > len(row):
            raise ValueError(f"Invalid padded action count: {count}")
        if count < min_count or count > max_count:
            raise ValueError(f"Padded action count {count} is outside select bounds [{min_count}, {max_count}].")
        action = [int(index) for index in row[:count]]
        if len(action) != len(set(action)):
            raise ValueError("Duplicate selected option indices are not allowed.")
        for index in action:
            if index < 0 or index >= len(mask) or not mask[index]:
                raise ValueError(f"Selected illegal option index: {index}")
        actions.append(action)
    return actions


def _assert_output_spec(out: dict[str, np.ndarray], config: EncoderConfig) -> None:
    specs = encoded_observation_spec(config)
    if set(out) != set(specs):
        raise ValueError("Encoded observation output keys do not match the encoder spec.")
    for key, field in specs.items():
        value = out[key]
        expected_dtype = np.dtype(numpy_dtype_for_encoded_key(key))
        if tuple(value.shape) != field.shape or value.dtype != expected_dtype:
            raise ValueError(
                f"Encoded observation output {key!r} has shape/dtype "
                f"{tuple(value.shape)}/{value.dtype}; expected {field.shape}/{expected_dtype}."
            )


def _assert_encoded_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    if set(left) != set(right):
        raise AssertionError("Encoded observation keys differ.")
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=f"Encoded observation field {key!r} differs.")


def _write_card_ids(out: np.ndarray, values, *, fill: int) -> None:
    out.fill(fill)
    for index, value in enumerate(values):
        if index >= out.shape[0]:
            break
        out[index] = int(value)


def _write_pokemon_attr(out: np.ndarray, pokemon, attr: str, *, fill: int) -> None:
    out.fill(fill)
    for index, value in enumerate(pokemon[: out.shape[0]]):
        out[index] = int(getattr(value, attr))


def _write_pokemon_len(out: np.ndarray, pokemon, attr: str) -> None:
    out.fill(0)
    for index, value in enumerate(pokemon[: out.shape[0]]):
        out[index] = len(getattr(value, attr))


def _write_active_energy_types(out: np.ndarray, players: list[Any]) -> None:
    out.fill(-1)
    for player_index, player in enumerate(players[: out.shape[0]]):
        if player.active is None:
            continue
        for energy_index, energy in enumerate(player.active.energies[: out.shape[1]]):
            out[player_index, energy_index] = int(energy)


def _write_active_tool_card_ids(out: np.ndarray, players: list[Any]) -> None:
    out.fill(-1)
    for player_index, player in enumerate(players[: out.shape[0]]):
        if player.active is None:
            continue
        for tool_index, tool in enumerate(player.active.tools[: out.shape[1]]):
            out[player_index, tool_index] = int(tool.id)


def _write_select_options(out: np.ndarray, select: Any | None) -> None:
    out.fill(-1)
    if select is None:
        return
    width = out.shape[1]
    for index, option in enumerate(select.option[: out.shape[0]]):
        row = (
            _none_to(option.type, -1),
            _none_to(option.number, -1),
            _none_to(option.area, -1),
            _none_to(option.index, -1),
            _none_to(option.playerIndex, -1),
            _none_to(option.toolIndex, -1),
            _none_to(option.energyIndex, -1),
            _none_to(option.count, -1),
            _none_to(option.inPlayArea, -1),
            _none_to(option.inPlayIndex, -1),
            _none_to(option.attackId, -1),
            _none_to(option.cardId, -1),
            _none_to(option.serial, -1),
            _none_to(option.specialConditionType, -1),
        )
        out[index, :width] = row[:width]


def _write_legal_action_mask(out: np.ndarray, select: Any | None) -> None:
    out.fill(False)
    count = 0 if select is None else min(len(select.option), out.shape[0])
    out[:count] = True


def _write_categorical_action_mask(out: np.ndarray, select: Any | None) -> None:
    out.fill(False)
    if select is None:
        return
    out[0] = select.minCount == 0
    if select.minCount <= 1 <= select.maxCount:
        option_slots = min(len(select.option), out.shape[0] - 1)
        out[1 : option_slots + 1] = True


def _write_selection_count_mask(out: np.ndarray, select: Any | None) -> None:
    out.fill(False)
    if select is None:
        return
    legal_options = len(select.option)
    min_count = int(select.minCount)
    max_count = min(int(select.maxCount), legal_options, out.shape[0] - 1)
    out[min_count : max_count + 1] = True


def _require_integral_values(values, name: str) -> None:
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"{name} must contain integer values.")


def _metadata_table_size(explicit_max_id: int | None, values: Any) -> int:
    inferred = max(values.keys(), default=0)
    if explicit_max_id is None:
        return inferred
    if explicit_max_id < 0:
        raise ValueError("metadata table max IDs must be non-negative.")
    return explicit_max_id


def _card_ids_by_name(card_db: CardDB) -> dict[str, tuple[int, ...]]:
    ids_by_name: dict[str, list[int]] = {}
    for card_id, card in card_db.cards.items():
        ids_by_name.setdefault(_metadata_name_key(card.name), []).append(card_id)
    return {name: tuple(sorted(ids)) for name, ids in ids_by_name.items()}


def _evolves_to_ids(card_db: CardDB, card_ids_by_name: dict[str, tuple[int, ...]]) -> dict[int, tuple[int, ...]]:
    evolves_to: dict[int, list[int]] = {}
    for evolved_id, card in card_db.cards.items():
        if not card.evolvesFrom:
            continue
        for source_id in card_ids_by_name.get(_metadata_name_key(card.evolvesFrom), ()):
            evolves_to.setdefault(source_id, []).append(evolved_id)
    return {card_id: tuple(sorted(ids)) for card_id, ids in evolves_to.items()}


def _metadata_name_key(value: str) -> str:
    return " ".join(value.lower().replace("é", "e").split())


def _raw_text(raw: Any, *aliases: str) -> str:
    if not raw:
        return ""
    by_key = {str(key).lower(): str(value).strip() for key, value in raw.items()}
    for alias in aliases:
        value = by_key.get(alias.lower(), "")
        if value.lower() not in {"", "n/a", "na", "none", "null", "-"}:
            return value
    return ""


def _stable_text_id(value: str) -> int:
    text = value.strip().lower()
    if not text:
        return 0
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _parse_metadata_int(value: str) -> int:
    match = re.search(r"-?\d+", value)
    return int(match.group(0)) if match else 0


def _encoder_config_to_dict(config: EncoderConfig) -> dict[str, int]:
    return {field.name: int(getattr(config, field.name)) for field in fields(config)}


def _field_spec_to_dict(spec: TensorFieldSpec) -> dict:
    return {"shape": list(spec.shape), "dtype": spec.dtype}


def _field_spec_mapping_to_dict(specs: dict[str, TensorFieldSpec]) -> dict[str, dict]:
    return {key: _field_spec_to_dict(spec) for key, spec in specs.items()}


def _schema_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_device(device=None):
    torch = _torch()
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _observation_samples(value, spec: TensorFieldSpec, key: str) -> torch.Tensor:
    torch = _torch()
    tensor = torch.as_tensor(value, dtype=torch.float32)
    feature_ndim = len(spec.shape)
    if tensor.ndim < feature_ndim:
        raise ValueError(f"tensor_obs[{key!r}] has fewer dimensions than its encoded feature shape.")
    if feature_ndim and tuple(tensor.shape[-feature_ndim:]) != spec.shape:
        raise ValueError(f"tensor_obs[{key!r}] does not match encoded feature shape {spec.shape}.")
    sample_shape = tensor.shape[: tensor.ndim - feature_ndim]
    sample_count = int(np.prod(sample_shape)) if sample_shape else 1
    return tensor.reshape(sample_count, *spec.shape)


def _torch():
    import torch

    return torch


def _observation_sample_count(
    tensor_obs: dict[str, Any],
    specs: dict[str, TensorFieldSpec],
    keys: tuple[str, ...],
) -> int:
    counts = []
    for key in keys:
        if key in tensor_obs and key in specs:
            counts.append(_observation_samples(tensor_obs[key], specs[key], key).shape[0])
    if not counts:
        return 0
    first = int(counts[0])
    if any(int(count) != first for count in counts):
        raise ValueError("All normalized observation keys must have the same sample count.")
    return first


def _active(player: dict[str, Any]) -> dict[str, Any] | None:
    return player["active"][0] if player["active"] else None


def _bench_ids(player: dict[str, Any], max_bench: int) -> list[int]:
    return [pokemon["id"] for pokemon in player["bench"][:max_bench]]


def _bench_hp(player: dict[str, Any], max_bench: int) -> list[int]:
    return [pokemon["hp"] for pokemon in player["bench"][:max_bench]]


def _bench_energy_counts(player: dict[str, Any], max_bench: int) -> list[int]:
    return [len(pokemon["energies"]) for pokemon in player["bench"][:max_bench]]


def _bench_tool_counts(player: dict[str, Any], max_bench: int) -> list[int]:
    return [len(pokemon["tools"]) for pokemon in player["bench"][:max_bench]]


def _active_energy_types(players: list[dict[str, Any]], max_attached_energy: int) -> list[list[int]]:
    return [
        _pad([] if (active := _active(player)) is None else [int(energy) for energy in active["energies"]], max_attached_energy, fill=-1)
        for player in players
    ]


def _active_tool_card_ids(players: list[dict[str, Any]], max_attached_tools: int) -> list[list[int]]:
    return [
        _pad([] if (active := _active(player)) is None else [tool["id"] for tool in active["tools"]], max_attached_tools, fill=-1)
        for player in players
    ]


def _stadium_card_id(current: dict[str, Any]) -> int:
    stadium = current.get("stadium") or []
    return -1 if not stadium else int(stadium[0]["id"])


def _bench_ids_from_state(player: Any, max_bench: int) -> list[int]:
    return [pokemon.id for pokemon in player.bench[:max_bench]]


def _bench_hp_from_state(player: Any, max_bench: int) -> list[int]:
    return [pokemon.hp for pokemon in player.bench[:max_bench]]


def _bench_energy_counts_from_state(player: Any, max_bench: int) -> list[int]:
    return [len(pokemon.energies) for pokemon in player.bench[:max_bench]]


def _bench_tool_counts_from_state(player: Any, max_bench: int) -> list[int]:
    return [len(pokemon.tools) for pokemon in player.bench[:max_bench]]


def _active_energy_types_from_state(players: list[Any], max_attached_energy: int) -> list[list[int]]:
    return [
        _pad([] if player.active is None else [int(energy) for energy in player.active.energies], max_attached_energy, fill=-1)
        for player in players
    ]


def _active_tool_card_ids_from_state(players: list[Any], max_attached_tools: int) -> list[list[int]]:
    return [
        _pad([] if player.active is None else [tool.id for tool in player.active.tools], max_attached_tools, fill=-1)
        for player in players
    ]


def _stadium_card_id_from_state(state: Any) -> int:
    return -1 if not state.stadium else int(state.stadium[0].id)


def _pad(values, size: int, fill: int) -> list[int]:
    values = list(values)
    return values[:size] + [fill] * max(0, size - len(values))


def _none_to(value: int | None, fill: int) -> int:
    return fill if value is None else int(value)

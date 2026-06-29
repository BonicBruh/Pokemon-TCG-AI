"""Fixed rule-based training opponents."""
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

@dataclass
class Opponent:
    name: str
    deck: list[int]
    act: Callable[[dict], list[int]]
    reset: Callable[[], None]


def _read_deck(path: Path) -> list[int]:
    deck = [int(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} contains {len(deck)} cards, expected 60")
    return deck


def load_opponent(name: str) -> Opponent:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "lucario": "mega_lucario",
        "mega_lucario_ex": "mega_lucario",
        "starmie": "mega_starmie",
        "mega_starmie_ex": "mega_starmie",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"mega_lucario", "mega_starmie"}:
        raise KeyError(f"Unknown opponent {name!r}")
    module = import_module(f"opponents.{normalized}.agent")
    deck = _read_deck(Path(module.__file__).with_name("deck.csv"))
    reset = getattr(module, "reset_state", lambda: None)
    return Opponent(normalized, deck, module.agent, reset)

__all__ = ["Opponent", "load_opponent"]

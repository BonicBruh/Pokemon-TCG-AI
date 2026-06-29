from __future__ import annotations
from collections import Counter
from pathlib import Path


def read_deck(path: str | Path) -> list[int]:
    path = Path(path)
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"Deck {path} has {len(deck)} cards; expected 60")
    invalid = {card: count for card, count in Counter(deck).items() if count > 4 and card not in range(1, 10)}
    if invalid:
        raise ValueError(f"Non-basic-Energy cards exceed four copies: {invalid}")
    return deck


def write_deck(path: str | Path, deck: list[int]) -> None:
    if len(deck) != 60:
        raise ValueError("A deck must contain exactly 60 cards")
    Path(path).write_text("\n".join(map(str, deck)) + "\n", encoding="utf-8")

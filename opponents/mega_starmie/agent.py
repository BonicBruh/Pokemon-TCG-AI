"""Kaggle entry point for the Mega Starmie rule-based agent."""

from __future__ import annotations

import os

from cg.api import Observation, to_observation_class

from .strategy import choose_action


def read_deck_csv() -> list[int]:
    """Load the 60-card deck from the submission bundle."""
    path = os.path.join(os.path.dirname(__file__), "deck.csv")
    if not os.path.exists(path):
        path = os.path.join("/kaggle_simulations/agent", "deck.csv")

    with open(path, "r", encoding="utf-8") as file:
        deck = [int(line.strip()) for line in file if line.strip()]

    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 card IDs, found {len(deck)}")
    return deck


def agent(obs_dict: dict) -> list[int]:
    """Return legal option indices for one simulator decision."""
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    try:
        return choose_action(obs)
    except Exception:
        # A valid conservative fallback is preferable to forfeiting a game if
        # the engine adds an unfamiliar selection context during competition.
        minimum = obs.select.minCount
        return list(range(minimum))


def reset_state() -> None:
    """The Starmie policy is stateless between games."""
    return None

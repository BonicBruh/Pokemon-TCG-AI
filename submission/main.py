from __future__ import annotations
import os
from pathlib import Path
import torch
from cg.api import to_observation_class
from ptcg_ppo.checkpoint import load_checkpoint
from ptcg_ppo.observations import encode_batch
from ptcg_ppo.policy import actions_to_lists, sample_actions

BASE=Path(__file__).resolve().parent
if not (BASE/"deck.csv").exists(): BASE=Path("/kaggle_simulations/agent")
_MODEL=None; _CONFIG=None; _DEVICE=torch.device("cpu")

def read_deck_csv():
    deck=[int(x) for x in (BASE/"deck.csv").read_text().splitlines() if x.strip()]
    if len(deck)!=60: raise ValueError(f"deck.csv has {len(deck)} cards")
    return deck

def _load():
    global _MODEL,_CONFIG
    if _MODEL is None:
        _MODEL,_CONFIG,_=load_checkpoint(BASE/"model.pt",_DEVICE); _MODEL.eval()

def agent(obs_dict):
    if obs_dict.get("select") is None: return read_deck_csv()
    try:
        _load()
        with torch.no_grad(): action=sample_actions(_MODEL,encode_batch([obs_dict],_CONFIG,_DEVICE),deterministic=True)
        return actions_to_lists(action.indices,action.counts)[0]
    except Exception:
        select=obs_dict["select"]; return list(range(int(select["minCount"])))

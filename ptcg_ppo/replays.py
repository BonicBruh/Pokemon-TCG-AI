from __future__ import annotations
import glob, json, os, random
from pathlib import Path
import torch
from torch.utils.data import IterableDataset
from tcg_rl.encoding import EncoderConfig, encode_observation_numpy


def replay_files(paths):
    files=[]
    for path in paths:
        p=Path(path)
        if p.is_file() and p.suffix.lower()==".json": files.append(str(p))
        elif p.is_dir(): files.extend(glob.glob(str(p/"**/*.json"), recursive=True))
    return sorted(set(files))


def _deck_from_episode(ep, player):
    try:
        visualize = ep["steps"][0][0]["visualize"][0]
        return list(map(int, visualize["action"][player]))
    except Exception:
        return None


class ReplayDecisionDataset(IterableDataset):
    """Streams legal observation/action pairs directly from Kaggle episode JSON."""
    def __init__(self, paths, encoder_config: EncoderConfig, *, winners_only=True, max_examples=None,
                 target_deck=None, seed=0):
        super().__init__(); self.files=replay_files(paths); self.config=encoder_config
        self.winners_only=winners_only; self.max_examples=max_examples
        self.target_deck=None if target_deck is None else sorted(map(int,target_deck)); self.seed=seed
        if not self.files: raise FileNotFoundError(f"No replay JSON files found in {paths}")

    def __iter__(self):
        info=torch.utils.data.get_worker_info(); worker_id=0 if info is None else info.id
        num_workers=1 if info is None else info.num_workers
        files=self.files[worker_id::num_workers]
        rng=random.Random(self.seed+worker_id); rng.shuffle(files)
        yielded=0
        for filename in files:
            try: ep=json.loads(Path(filename).read_text(encoding="utf-8"))
            except Exception: continue
            rewards=ep.get("rewards",[0,0])
            for player in (0,1):
                if self.winners_only and (player>=len(rewards) or rewards[player] <= 0): continue
                if self.target_deck is not None:
                    deck=_deck_from_episode(ep,player)
                    if deck is None or sorted(deck)!=self.target_deck: continue
                for step in ep.get("steps",[]):
                    if player>=len(step): continue
                    entry=step[player]; obs=entry.get("observation") or {}; action=entry.get("action")
                    current=obs.get("current"); select=obs.get("select")
                    if current is None or select is None or action is None: continue
                    if int(current.get("result",-1)) >= 0 or int(current.get("yourIndex",-1)) != player: continue
                    if entry.get("status") not in {"ACTIVE","DONE"}: continue
                    action=list(map(int,action)); n=len(select.get("option",[]))
                    if not (int(select["minCount"]) <= len(action) <= int(select["maxCount"])): continue
                    if len(set(action)) != len(action) or any(x<0 or x>=n for x in action): continue
                    encoded=encode_observation_numpy(obs,self.config)
                    padded=torch.full((self.config.max_options,),-1,dtype=torch.long)
                    if action: padded[:len(action)]=torch.tensor(action,dtype=torch.long)
                    yield {**{k:torch.from_numpy(v) for k,v in encoded.items()},
                           "chosen_indices":padded,"chosen_count":torch.tensor(len(action),dtype=torch.long)}
                    yielded += 1
                    if self.max_examples is not None and yielded>=self.max_examples: return

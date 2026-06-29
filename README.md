# Pokémon TCG AI — PPO vs Mega Lucario and Mega Starmie

This repository trains a **masked multi-select PPO agent** on the real `cg`/CABT simulator. The learner uses the supplied 60-card Mega Kangaskhan multitype deck and samples training opponents from the attached rule-based **Mega Lucario ex** and **Mega Starmie ex** agents.

The policy keeps the supplied architecture:

- `tcg_rl/network/encoder.py`: fixed-shape observation encoder
- `tcg_rl/network/actor_critic.py`: encoder + multi-select policy head + value head
- `tcg_rl/encoding.py`: CABT observation/options encoder and legal masks

The repository contains training code, the simulator bindings, both fixed opponents, replay behavior-cloning support, evaluation, and submission export. It does **not** pretend that an untrained checkpoint is competitive: train and evaluate it before submitting.

## Deck

The simulator ID deck is in `decks/kangaskhan_multitype.csv`. The corrected Telepath Energy entry is ID `19` (`Telepath Psychic Energy`, POR 87), not POR 88.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-training.txt
python validate_deck.py
python smoke_test.py
```

`cg/libcg.so` is used on Linux and `cg/cg.dll` on Windows.

## Replay data

`data/manifest.csv` lists the daily public episode datasets. Most daily archives are very large, so the downloader refuses to fetch tens of gigabytes accidentally.

To select the latest manifest entry and print its size:

```bash
python download_episodes.py --manifest data/manifest.csv --latest 1 --output data/episodes
```

List the first page of files without downloading:

```bash
python download_episodes.py --manifest data/manifest.csv --latest 1 --list-only
```

Download up to 100 matching JSON/archive files from that page:

```bash
python download_episodes.py --manifest data/manifest.csv --latest 1 --max-files 100 --output data/episodes
```

To download a whole daily dataset:

```bash
python download_episodes.py --manifest data/manifest.csv --latest 1 --output data/episodes --whole-dataset
```

On Kaggle, attaching the daily episode dataset as a Notebook input is normally preferable to downloading it again. Point `--bc-episodes` at the mounted folder.

## Train

Sparse PPO only:

```bash
python train.py \
  --num-envs 8 \
  --updates 500 \
  --rollout-steps 128 \
  --opponents mega_lucario,mega_starmie \
  --save-dir models
```

Warm-start on public replay decisions, then continue PPO:

```bash
python train.py \
  --bc-episodes data/episodes \
  --bc-winners-only \
  --bc-max-examples 100000 \
  --bc-epochs 1 \
  --num-envs 8 \
  --updates 500 \
  --save-dir models
```

Do not set `--bc-target-deck-only` unless the replay dataset actually contains this exact 60-card deck. Otherwise the behavior-cloning stream will be empty.

The learner seat is randomized per episode. Rewards are terminal win/loss plus a small Prize-card delta reward; legal actions are masked before sampling.

## Evaluate properly

Ten games are too noisy for deck claims. Use at least 100 balanced games per opponent:

```bash
python evaluate.py \
  --checkpoint models/latest.pt \
  --games 100 \
  --opponents mega_lucario,mega_starmie \
  --output evaluation.csv
```

Evaluation alternates player slots and reports a 95% Wilson interval.

## Export a Kaggle submission

```bash
python export_submission.py \
  --checkpoint models/latest.pt \
  --deck decks/kangaskhan_multitype.csv \
  --output submission.tar.gz
```

The generated archive has `main.py`, `deck.csv`, and `model.pt` at the top level, plus the runtime packages and CABT library.

## Kaggle Notebook command

```bash
!git clone https://github.com/BonicBruh/Pokemon-TCG-AI.git
%cd Pokemon-TCG-AI
!pip install -r requirements-training.txt
!python validate_deck.py
!python train.py --bc-episodes /kaggle/input/YOUR_EPISODE_DATASET --bc-winners-only --num-envs 8 --updates 500
!python evaluate.py --checkpoint models/latest.pt --games 100
!python export_submission.py --checkpoint models/latest.pt
```

## Important limitations

- Fixed-opponent training can overfit. Keep both opponents in the pool, evaluate against unseen replay agents, and later add self-play or a larger opponent pool.
- Public replay behavior cloning teaches general CABT action selection; it is not guaranteed to match this deck unless the same deck appears in the replay data.
- The CABT native library controls game randomness; it does not expose a per-game seed through the supplied Python wrapper.
- A model that beats these two rule agents may still perform poorly on the live leaderboard.

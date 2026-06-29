# Build verification

The repository was checked against the bundled CABT engine before packaging.

Verified:

- The 60-card Mega Kangaskhan multitype deck is accepted by CABT.
- Both supplied rule agents load and finish games without illegal selections.
- The fixed-shape encoder and masked multi-select policy produce legal actions.
- A two-worker PPO rollout and one optimizer update completed successfully.
- Replay JSON decisions from a Kaggle episode export were parsed into fixed-shape behavior-cloning batches.
- Submission export placed `main.py`, `deck.csv`, and `model.pt` at archive root and loaded the exported model in a complete CABT game.
- Unit tests passed.

No competitive trained weights are included. `models/.gitkeep` is only a placeholder; run training before exporting a real submission.

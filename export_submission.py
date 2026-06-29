from __future__ import annotations
import argparse, shutil, tarfile, tempfile
from pathlib import Path
from ptcg_ppo.decks import read_deck

RUNTIME=["cg","tcg_rl","ptcg_ppo"]
EXCLUDE={"env.py","vec_env.py","replays.py","bc.py","ppo.py"}

def copy_runtime(src,dst):
    if src.is_dir():
        shutil.copytree(src,dst,ignore=lambda path,names:{n for n in names if n in {"__pycache__"} or n.endswith(".pyc") or n in EXCLUDE})
    else: shutil.copy2(src,dst)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--deck",default="decks/kangaskhan_multitype.csv")
    p.add_argument("--output",default="submission.tar.gz"); args=p.parse_args(); root=Path(__file__).resolve().parent
    read_deck(args.deck)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); shutil.copy2(root/"submission/main.py",td/"main.py"); shutil.copy2(args.deck,td/"deck.csv"); shutil.copy2(args.checkpoint,td/"model.pt")
        for item in RUNTIME: copy_runtime(root/item,td/item)
        with tarfile.open(args.output,"w:gz") as tar:
            for path in sorted(td.rglob("*")):
                if path.is_file(): tar.add(path,arcname=str(path.relative_to(td)))
    with tarfile.open(args.output,"r:gz") as tar:
        names=tar.getnames(); assert "main.py" in names and "deck.csv" in names and "model.pt" in names
    print(f"Created {args.output} with {len(names)} files")
if __name__=="__main__": main()

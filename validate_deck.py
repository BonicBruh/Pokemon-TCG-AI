from __future__ import annotations
import argparse
from cg.game import battle_finish,battle_start
from ptcg_ppo.decks import read_deck

def main():
    p=argparse.ArgumentParser();p.add_argument("deck",nargs="?",default="decks/kangaskhan_multitype.csv");a=p.parse_args()
    deck=read_deck(a.deck);obs,start=battle_start(deck,deck)
    try:
        if obs is None: raise SystemExit(f"CABT rejected deck: player={start.errorPlayer} error={start.errorType}")
        print(f"OK: {a.deck} contains 60 cards and CABT accepted it")
    finally:
        if obs is not None:battle_finish()
if __name__=="__main__":main()

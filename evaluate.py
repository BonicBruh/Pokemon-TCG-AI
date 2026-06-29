from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import torch
from ptcg_ppo.checkpoint import load_checkpoint
from ptcg_ppo.decks import read_deck
from ptcg_ppo.env import CabtMatchEnv
from ptcg_ppo.observations import encode_batch
from ptcg_ppo.policy import actions_to_lists, sample_actions


def wilson(wins,n,z=1.96):
    if n==0:return (0.0,1.0)
    p=wins/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h


def play(model,config,deck,opponent,seat,device):
    env=CabtMatchEnv(deck,(opponent,),learner_slot=seat,seed=seat)
    obs,_=env.reset(opponent_name=opponent,learner_slot=seat); total=0
    try:
        while True:
            with torch.no_grad(): action=sample_actions(model,encode_batch([obs],config,device),deterministic=True)
            obs,r,term,trunc,info=env.step(actions_to_lists(action.indices,action.counts)[0]); total+=r
            if term or trunc:return info,total
    finally: env.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--deck",default="decks/kangaskhan_multitype.csv")
    p.add_argument("--games",type=int,default=50); p.add_argument("--opponents",default="mega_lucario,mega_starmie")
    p.add_argument("--device",default="auto"); p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--output",default="evaluation.csv"); args=p.parse_args()
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    torch.set_num_threads(max(1,args.torch_threads))
    model,config,_=load_checkpoint(args.checkpoint,device); model.eval(); deck=read_deck(args.deck); rows=[]
    for opponent in [x.strip() for x in args.opponents.split(",") if x.strip()]:
        wins=0
        for game in range(args.games):
            seat=game%2; info,total=play(model,config,deck,opponent,seat,device)
            win=info["winner"]==seat; wins+=win
            rows.append({"opponent":opponent,"game":game,"learner_slot":seat,"winner":info["winner"],"win":int(win),"reward":total,
                         "learner_decisions":info["learner_decisions"],"engine_decisions":info["total_engine_decisions"]})
        low,high=wilson(wins,args.games); print(f"{opponent}: {wins}/{args.games} = {wins/args.games:.1%} (95% Wilson {low:.1%}-{high:.1%})")
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__=="__main__": main()

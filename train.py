from __future__ import annotations
import argparse, json, os, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tcg_rl.encoding import EncoderConfig
from ptcg_ppo.bc import behavior_clone
from ptcg_ppo.checkpoint import DEFAULT_MODEL_CONFIG, build_model, save_checkpoint
from ptcg_ppo.decks import read_deck
from ptcg_ppo.ppo import PPOConfig, collect_rollout, ppo_update
from ptcg_ppo.replays import ReplayDecisionDataset
from ptcg_ppo.vec_env import SubprocCabtVecEnv


def parse_args():
    p=argparse.ArgumentParser(description="Train PPO against Mega Lucario and Mega Starmie rule agents")
    p.add_argument("--deck",default="decks/kangaskhan_multitype.csv")
    p.add_argument("--opponents",default="mega_lucario,mega_starmie")
    p.add_argument("--num-envs",type=int,default=8); p.add_argument("--updates",type=int,default=500)
    p.add_argument("--rollout-steps",type=int,default=128); p.add_argument("--update-epochs",type=int,default=4)
    p.add_argument("--minibatch-size",type=int,default=512); p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--hidden-size",type=int,default=128); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="auto"); p.add_argument("--torch-threads",type=int,default=4)
    p.add_argument("--save-dir",default="models")
    p.add_argument("--save-every",type=int,default=25); p.add_argument("--resume",default=None)
    p.add_argument("--bc-episodes",nargs="*",default=[]); p.add_argument("--bc-epochs",type=int,default=1)
    p.add_argument("--bc-max-examples",type=int,default=100000); p.add_argument("--bc-winners-only",action="store_true")
    p.add_argument("--bc-target-deck-only",action="store_true")
    p.add_argument("--start-method",default="spawn",choices=["spawn","forkserver"])
    return p.parse_args()


def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    torch.set_num_threads(max(1,args.torch_threads))
    deck=read_deck(args.deck); encoder_config=EncoderConfig(max_options=128)
    model_config={**DEFAULT_MODEL_CONFIG,"hidden_size":args.hidden_size,"max_options":encoder_config.max_options}
    start_update=0
    if args.resume:
        payload=torch.load(args.resume,map_location=device,weights_only=False)
        model_config=dict(payload["model_config"])
        encoder_config=EncoderConfig(**payload["encoder_config"])
        model=build_model(model_config,device); model.load_state_dict(payload["model_state_dict"])
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
        if payload.get("optimizer_state_dict"): optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_update=int(payload.get("step",0))
    else:
        model=build_model(model_config,device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    if args.bc_episodes:
        dataset=ReplayDecisionDataset(args.bc_episodes,encoder_config,winners_only=args.bc_winners_only,
            max_examples=args.bc_max_examples,target_deck=deck if args.bc_target_deck_only else None,seed=args.seed)
        behavior_clone(model,dataset,device=device,epochs=args.bc_epochs,batch_size=128,learning_rate=1e-4)
    ppo_config=PPOConfig(rollout_steps=args.rollout_steps,update_epochs=args.update_epochs,minibatch_size=args.minibatch_size)
    env_kwargs={"learner_deck":deck,"opponent_names":tuple(x.strip() for x in args.opponents.split(",") if x.strip()),"seed":args.seed}
    save_dir=Path(args.save_dir); save_dir.mkdir(parents=True,exist_ok=True)
    history=[]; t0=time.time()
    with SubprocCabtVecEnv(args.num_envs,env_kwargs,start_method=args.start_method) as envs:
        observations,_=envs.reset()
        for update in range(start_update+1,start_update+args.updates+1):
            batch,observations,episodes=collect_rollout(model,envs,observations,encoder_config,device,ppo_config)
            metrics=ppo_update(model,optimizer,batch,device,ppo_config)
            wins=sum(1 for e in episodes if e.get("winner")==e.get("learner_slot")); games=len(episodes)
            row={"update":update,"games":games,"wins":wins,"win_rate":wins/games if games else None,**metrics,
                 "elapsed_seconds":time.time()-t0}
            history.append(row); print(json.dumps(row))
            with (save_dir/"history.jsonl").open("a",encoding="utf-8") as history_file:
                history_file.write(json.dumps(row)+"\n")
            if update%args.save_every==0 or update==start_update+args.updates:
                save_checkpoint(save_dir/"latest.pt",model,optimizer,encoder_config=encoder_config,
                    model_config=model_config,step=update,extra={"args":vars(args),"last_metrics":row})

if __name__=="__main__": main()

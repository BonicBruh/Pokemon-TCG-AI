from __future__ import annotations
import tempfile
import torch
from tcg_rl.encoding import EncoderConfig
from ptcg_ppo.checkpoint import DEFAULT_MODEL_CONFIG,build_model,save_checkpoint
from ptcg_ppo.decks import read_deck
from ptcg_ppo.env import CabtMatchEnv
from ptcg_ppo.observations import encode_batch
from ptcg_ppo.policy import actions_to_lists,sample_actions

def main():
    torch.set_num_threads(1)
    deck=read_deck("decks/kangaskhan_multitype.csv"); config=EncoderConfig(max_options=128)
    model=build_model({**DEFAULT_MODEL_CONFIG,"hidden_size":64}); model.eval()
    for opponent in ("mega_lucario","mega_starmie"):
        env=CabtMatchEnv(deck,(opponent,),seed=7,max_learner_decisions=50)
        obs,_=env.reset(opponent_name=opponent,learner_slot=0)
        try:
            for _ in range(50):
                with torch.no_grad():a=sample_actions(model,encode_batch([obs],config),deterministic=False)
                obs,_,term,trunc,info=env.step(actions_to_lists(a.indices,a.counts)[0])
                if term or trunc:break
            print(opponent,info)
        finally:env.close()
    print("Smoke test passed")
if __name__=="__main__":main()

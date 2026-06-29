from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .observations import encode_batch
from .policy import actions_to_lists, evaluate_actions, sample_actions

@dataclass
class PPOConfig:
    rollout_steps:int=128; update_epochs:int=4; minibatch_size:int=512
    gamma:float=0.995; gae_lambda:float=0.95; clip_coef:float=0.2
    ent_coef:float=0.01; vf_coef:float=0.5; max_grad_norm:float=0.5
    target_kl:float|None=0.03


def _stack_obs(rows):
    keys=rows[0].keys(); return {k:torch.stack([x[k] for x in rows],dim=0) for k in keys}


def collect_rollout(model, envs, observations, encoder_config, device, config:PPOConfig):
    obs_buf=[]; action_buf=[]; count_buf=[]; logp_buf=[]; value_buf=[]; reward_buf=[]; done_buf=[]
    episode_infos=[]
    model.eval()
    for _ in range(config.rollout_steps):
        obs_t=encode_batch(observations,encoder_config,device)
        with torch.no_grad(): action=sample_actions(model,obs_t)
        env_actions=actions_to_lists(action.indices,action.counts)
        next_obs,rewards,terminated,truncated,infos=envs.step(env_actions)
        dones=[a or b for a,b in zip(terminated,truncated,strict=True)]
        obs_buf.append({k:v.detach().cpu() for k,v in obs_t.items()})
        action_buf.append(action.indices.detach().cpu()); count_buf.append(action.counts.detach().cpu())
        logp_buf.append(action.log_probs.detach().cpu()); value_buf.append(action.values.detach().cpu())
        reward_buf.append(torch.tensor(rewards,dtype=torch.float32)); done_buf.append(torch.tensor(dones,dtype=torch.bool))
        for i,done in enumerate(dones):
            if done: episode_infos.append({**infos[i],"episode_reward_terminal":rewards[i]})
        reset_indices=[i for i,d in enumerate(dones) if d]
        if reset_indices:
            reset=envs.reset_at(reset_indices)
            for i,(fresh,_) in reset.items(): next_obs[i]=fresh
        observations=next_obs
    with torch.no_grad():
        next_values=model(encode_batch(observations,encoder_config,device))["values"].cpu()
    rewards=torch.stack(reward_buf); dones=torch.stack(done_buf); values=torch.stack(value_buf)
    advantages=torch.zeros_like(rewards); last=torch.zeros(rewards.shape[1])
    for t in range(config.rollout_steps-1,-1,-1):
        nv=next_values if t==config.rollout_steps-1 else values[t+1]
        nonterminal=(~dones[t]).float()
        delta=rewards[t]+config.gamma*nv*nonterminal-values[t]
        last=delta+config.gamma*config.gae_lambda*nonterminal*last; advantages[t]=last
    returns=advantages+values
    batch={k:torch.cat([row[k] for row in obs_buf],dim=0) for k in obs_buf[0]}
    batch.update({
        "chosen_indices":torch.cat(action_buf),"chosen_count":torch.cat(count_buf),
        "old_log_probs":torch.cat(logp_buf),"old_values":values.flatten(),
        "advantages":advantages.flatten(),"returns":returns.flatten(),
    })
    return batch,observations,episode_infos


def ppo_update(model,optimizer,batch,device,config:PPOConfig):
    model.train(); n=batch["advantages"].numel(); indices=np.arange(n)
    advantages=batch["advantages"]
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    batch={**batch,"advantages":advantages}
    metrics=[]
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0,n,config.minibatch_size):
            idx=torch.as_tensor(indices[start:start+config.minibatch_size],dtype=torch.long)
            obs={k:v[idx].to(device) for k,v in batch.items() if k not in {"chosen_indices","chosen_count","old_log_probs","old_values","advantages","returns"}}
            chosen=batch["chosen_indices"][idx].to(device); counts=batch["chosen_count"][idx].to(device)
            old_logp=batch["old_log_probs"][idx].to(device); old_values=batch["old_values"][idx].to(device)
            adv=batch["advantages"][idx].to(device); returns=batch["returns"][idx].to(device)
            logp,entropy,values,_=evaluate_actions(model,obs,chosen,counts)
            logratio=logp-old_logp; ratio=logratio.exp()
            pg1=-adv*ratio; pg2=-adv*torch.clamp(ratio,1-config.clip_coef,1+config.clip_coef)
            policy_loss=torch.maximum(pg1,pg2).mean()
            value_pred_clipped=old_values+(values-old_values).clamp(-config.clip_coef,config.clip_coef)
            value_loss=0.5*torch.maximum((values-returns).pow(2),(value_pred_clipped-returns).pow(2)).mean()
            entropy_loss=entropy.mean(); loss=policy_loss+config.vf_coef*value_loss-config.ent_coef*entropy_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm); optimizer.step()
            with torch.no_grad():
                approx_kl=((ratio-1)-logratio).mean(); clipfrac=((ratio-1).abs()>config.clip_coef).float().mean()
            metrics.append((loss.item(),policy_loss.item(),value_loss.item(),entropy_loss.item(),approx_kl.item(),clipfrac.item(),float(grad)))
        if config.target_kl is not None and metrics[-1][4]>config.target_kl: break
    m=np.asarray(metrics).mean(axis=0)
    return dict(zip(["loss","policy_loss","value_loss","entropy","approx_kl","clipfrac","grad_norm"],map(float,m),strict=True))

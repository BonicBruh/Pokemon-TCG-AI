from __future__ import annotations
import math
import torch
from torch.utils.data import DataLoader
from .policy import evaluate_actions


def behavior_clone(model, dataset, *, device, epochs=1, batch_size=128, learning_rate=1e-4,
                   num_workers=0, max_grad_norm=1.0, log_every=100):
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    model.train(); step=0; last={}
    for epoch in range(epochs):
        loader=DataLoader(dataset,batch_size=batch_size,num_workers=num_workers)
        for batch in loader:
            chosen=batch.pop("chosen_indices").to(device)
            counts=batch.pop("chosen_count").to(device)
            obs={k:v.to(device) for k,v in batch.items()}
            logp,entropy,_,_=evaluate_actions(model,obs,chosen,counts)
            loss=-logp.mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
            optimizer.step(); step+=1
            last={"bc_loss":float(loss.item()),"bc_entropy":float(entropy.mean().item()),"bc_grad_norm":float(grad)}
            if step%log_every==0: print(f"bc step={step} loss={last['bc_loss']:.4f} entropy={last['bc_entropy']:.3f}")
    return last

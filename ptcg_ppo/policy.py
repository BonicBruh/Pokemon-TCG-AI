from __future__ import annotations
from dataclasses import dataclass
import torch

NEG_INF = -1e9

@dataclass
class ActionBatch:
    indices: torch.Tensor
    counts: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("Every row must have at least one legal entry")
    return logits.masked_fill(~mask, NEG_INF)


def sample_actions(model, obs: dict[str, torch.Tensor], deterministic: bool = False) -> ActionBatch:
    out = model(obs)
    option_logits = out["option_logits"]
    count_logits = out["count_logits"]
    action_mask = obs["action_mask"].bool()
    count_mask = obs["selection_count_mask"].bool()
    batch, max_options = option_logits.shape

    masked_counts = _masked_logits(count_logits, count_mask)
    count_dist = torch.distributions.Categorical(logits=masked_counts)
    counts = masked_counts.argmax(dim=-1) if deterministic else count_dist.sample()
    log_probs = count_dist.log_prob(counts)
    entropy = count_dist.entropy()

    indices = torch.full((batch, max_options), -1, dtype=torch.long, device=option_logits.device)
    available = action_mask.clone()
    max_count = int(counts.max().item()) if counts.numel() else 0
    for position in range(max_count):
        active = counts > position
        if not bool(active.any()):
            break
        safe_available = available.clone()
        safe_available[~active, 0] = True
        masked_options = _masked_logits(option_logits, safe_available)
        dist = torch.distributions.Categorical(logits=masked_options)
        selected = masked_options.argmax(dim=-1) if deterministic else dist.sample()
        indices[active, position] = selected[active]
        log_probs = log_probs + torch.where(active, dist.log_prob(selected), torch.zeros_like(log_probs))
        entropy = entropy + torch.where(active, dist.entropy(), torch.zeros_like(entropy))
        available[active, selected[active]] = False

    return ActionBatch(indices, counts, log_probs, entropy, out["values"])


def evaluate_actions(model, obs: dict[str, torch.Tensor], indices: torch.Tensor, counts: torch.Tensor):
    out = model(obs)
    option_logits = out["option_logits"]
    count_logits = out["count_logits"]
    action_mask = obs["action_mask"].bool()
    count_mask = obs["selection_count_mask"].bool()

    count_dist = torch.distributions.Categorical(logits=_masked_logits(count_logits, count_mask))
    log_probs = count_dist.log_prob(counts)
    entropy = count_dist.entropy()
    available = action_mask.clone()
    max_count = int(counts.max().item()) if counts.numel() else 0
    for position in range(max_count):
        active = counts > position
        selected = indices[:, position]
        if torch.any(active & (selected < 0)):
            raise ValueError("Stored action is missing a required option index")
        safe_available = available.clone()
        safe_available[~active, 0] = True
        dist = torch.distributions.Categorical(logits=_masked_logits(option_logits, safe_available))
        safe_selected = selected.clamp_min(0)
        chosen_legal = available.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
        if torch.any(active & ~chosen_legal):
            raise ValueError("Stored action contains an illegal or duplicate option")
        log_probs = log_probs + torch.where(active, dist.log_prob(safe_selected), torch.zeros_like(log_probs))
        entropy = entropy + torch.where(active, dist.entropy(), torch.zeros_like(entropy))
        available[active, safe_selected[active]] = False
    return log_probs, entropy, out["values"], out


def actions_to_lists(indices: torch.Tensor, counts: torch.Tensor) -> list[list[int]]:
    idx = indices.detach().cpu().tolist()
    cnt = counts.detach().cpu().tolist()
    return [[int(x) for x in row[:int(n)]] for row, n in zip(idx, cnt, strict=True)]

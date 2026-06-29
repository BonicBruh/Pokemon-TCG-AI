"""Policy and value heads for actor-critic models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MultiSelectPolicyHead(torch.nn.Module):
    """Score CABT-style multi-select option sets from encoded board state.

    The head emits per-option inclusion preferences plus a separate cardinality
    distribution. Rollout code then samples a legal count and selects that many
    options without replacement, matching prompts where CABT expects
    ``list[int]`` selections instead of a single categorical action.
    """

    def __init__(self, hidden_size: int, max_options: int, *, mode: str = "active"):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if max_options <= 0:
            raise ValueError("max_options must be positive.")
        if mode not in {"active", "full"}:
            raise ValueError("policy head mode must be 'active' or 'full'.")
        self.max_options = max_options
        self.mode = mode
        self.option_query = torch.nn.Linear(hidden_size, hidden_size)
        self.option_key = torch.nn.Linear(hidden_size, hidden_size)
        self.option_selector = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.selection_count_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, max_options + 1),
        )

    def forward(
        self,
        board: torch.Tensor,
        options: torch.Tensor,
        *,
        profiler=None,
        profiler_prefix: str = "",
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if options.shape[:-2] != board.shape[:-1]:
            raise ValueError("options batch shape must match board batch shape.")
        if options.shape[-2] != self.max_options:
            raise ValueError("options max_options must match the policy head max_options.")
        if options.shape[-1] != board.shape[-1]:
            raise ValueError("options embedding size must match board embedding size.")
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in {"active", "full"}:
            raise ValueError("policy head mode must be 'active' or 'full'.")

        with _profile(profiler, f"{profiler_prefix}policy_head_input_prepare"):
            scale = board.shape[-1] ** 0.5
        with _profile(profiler, f"{profiler_prefix}policy_head_active_type_extract"):
            # Current architecture has one dense multi-select action head. There
            # is no per-prompt head id to route, so every row uses head 0.
            pass
        with _profile(profiler, f"{profiler_prefix}policy_head_all_heads"):
            query = self.option_query(board).unsqueeze(-2)
            keys = self.option_key(options)
            inclusion_logits = (query * keys).sum(dim=-1) / scale
        with _profile(profiler, f"{profiler_prefix}policy_head_active_routing"):
            pass
        with _profile(profiler, f"{profiler_prefix}policy_head_per_head_forward"):
            if selected_mode == "full":
                selector_logits = self._selector_logits_full(board, options)
            else:
                selector_logits = self._selector_logits_active(board, options)
            count_logits = self.selection_count_head(board)
        with _profile(profiler, f"{profiler_prefix}policy_head_concat_or_pack"):
            option_logits = inclusion_logits + selector_logits
        with _profile(profiler, f"{profiler_prefix}policy_head_scatter"):
            pass
        with _profile(profiler, f"{profiler_prefix}policy_head_mask_or_postprocess"):
            pass
        with _profile(profiler, f"{profiler_prefix}policy_head_output_alloc"):
            output = {
                "option_logits": option_logits,
                "count_logits": count_logits,
            }
            output["num_action_heads"] = torch.as_tensor(1, device=board.device)
            output["dense_logits_allocated"] = torch.as_tensor(True, device=board.device)
            output["compact_policy_output_enabled"] = torch.as_tensor(False, device=board.device)
        return output

    def _selector_logits_full(self, board: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        board_context = board.unsqueeze(-2).expand(*options.shape[:-1], board.shape[-1])
        return self.option_selector(torch.cat((board_context, options), dim=-1)).squeeze(-1)

    def _selector_logits_active(self, board: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        first = self.option_selector[0]
        activation = self.option_selector[1]
        final = self.option_selector[2]
        hidden_size = board.shape[-1]
        board_weight = first.weight[:, :hidden_size]
        option_weight = first.weight[:, hidden_size:]
        hidden = F.linear(options, option_weight, first.bias)
        hidden = hidden + F.linear(board, board_weight, None).unsqueeze(-2)
        return final(activation(hidden)).squeeze(-1)


def _profile(profiler, name: str):
    if profiler is None:
        from contextlib import nullcontext

        return nullcontext()
    return profiler.time(name)


class ValueHead(torch.nn.Module):
    """Estimate the scalar value of an encoded state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.value_head(board).squeeze(-1)


__all__ = ["MultiSelectPolicyHead", "ValueHead"]

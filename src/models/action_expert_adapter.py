from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Residual bottleneck adapter: x + W2(GELU(W1(x)))."""

    def __init__(self, dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.nn.functional.gelu(self.down(x)))


class ActionExpertAdapterBank(nn.Module):
    """One adapter per ActionDiT DiTBlock (applied to block output tokens)."""

    def __init__(self, action_expert: nn.Module, bottleneck_dim: int = 256):
        super().__init__()
        if not hasattr(action_expert, "blocks"):
            raise AttributeError("action_expert has no `blocks` ModuleList")
        hidden_dim = int(action_expert.hidden_dim)
        self.adapters = nn.ModuleList(
            [BottleneckAdapter(hidden_dim, bottleneck_dim) for _ in action_expert.blocks]
        )
        self._handles: list = []

    def attach(self, action_expert: nn.Module) -> None:
        self.detach()

        def _make_hook(adapter: BottleneckAdapter):
            def _hook(_module, _inputs, output):
                return adapter(output)

            return _hook

        for block, adapter in zip(action_expert.blocks, self.adapters):
            self._handles.append(block.register_forward_hook(_make_hook(adapter)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def state_dict_adapter_only(self) -> dict[str, torch.Tensor]:
        return self.state_dict()

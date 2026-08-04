"""Stateful module API.

Each :class:`StatefulModule` exposes :meth:`init_state` returning a dict of
per-module tensors. :func:`init_states` walks an ``nn.Module`` tree, calls
``init_state`` on every stateful submodule, and returns a ``dict[name -> state]``
that callers thread through ``forward`` via a ``model_state`` argument.

State is mutated only by :meth:`increment_step` (called explicitly via
:func:`increment_steps`) and by ``forward`` writing into preallocated buffers
at known offsets.  No magic context manager, no implicit per-module storage.
"""

from abc import ABC, abstractmethod
from copy import copy
from typing import Any
import torch
from torch import nn


State = dict[str, Any]
ModelState = dict[str, State]


def _clone_state_value(value: Any, *, detach: bool) -> Any:
    """Clone every mutable value used by streaming inference state.

    ``mamba_ssm.InferenceParams`` is intentionally handled without importing
    the optional CUDA package.  A shallow copy of the small Python container
    is safe only after all of its attributes (including the per-layer tensor
    cache) have been recursively cloned.
    """

    if isinstance(value, torch.Tensor):
        value = value.detach() if detach else value
        return value.clone()
    if isinstance(value, dict):
        return {
            key: _clone_state_value(item, detach=detach)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item, detach=detach) for item in value)
    if isinstance(value, list):
        return [_clone_state_value(item, detach=detach) for item in value]
    if hasattr(value, "key_value_memory_dict"):
        cloned = copy(value)
        for name, item in vars(value).items():
            setattr(cloned, name, _clone_state_value(item, detach=detach))
        return cloned
    # Offsets, maximum lengths, booleans, None, and immutable metadata.
    return value


def clone_model_state(
    model_state: ModelState,
    *,
    detach: bool = True,
) -> ModelState:
    """Return a storage-independent snapshot of a complete model state.

    Clean Acoustic Cache relies on a transactional fork: event decoding may
    mutate its private state, while the persistent audio state must remain
    bit-identical.  ``dict.copy`` is therefore never sufficient here.
    """

    return _clone_state_value(model_state, detach=detach)


def iter_state_tensors(model_state: ModelState):
    """Yield ``(path, tensor)`` pairs for diagnostics and isolation tests."""

    def visit(value: Any, path: str):
        if isinstance(value, torch.Tensor):
            yield path, value
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                yield from visit(value[key], f"{path}.{key}" if path else str(key))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from visit(item, f"{path}[{index}]")
        else:
            cache = getattr(value, "key_value_memory_dict", None)
            if cache is not None:
                yield from visit(cache, f"{path}.key_value_memory_dict")
            lengths = getattr(value, "lengths_per_sample", None)
            if isinstance(lengths, torch.Tensor):
                yield f"{path}.lengths_per_sample", lengths

    yield from visit(model_state, "")


class StatefulModule(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_absolute_name: str | None = None

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int) -> State:
        raise NotImplementedError

    def increment_step(self, state: State, increment: int = 1) -> None:
        pass

    def reorder_state(self, state: State, indices: torch.Tensor) -> None:
        """Reorder a batched inference state after beam selection.

        Stateful modules with a non-standard batch dimension should override
        this method.  The conservative default handles tensors whose leading
        dimension is the beam batch and leaves metadata/scalars untouched.
        """
        if indices.numel() == 0:
            return
        required = int(indices.max().item()) + 1
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] >= required:
                    state[key] = value.index_select(0, indices.to(value.device))

    def get_state(self, model_state: ModelState | None) -> State | None:
        if model_state is None or self._module_absolute_name is None:
            return None
        return model_state.get(self._module_absolute_name)


def init_states(model: nn.Module, batch_size: int, sequence_length: int) -> ModelState:
    """Allocate state for every :class:`StatefulModule` reachable from ``model``.

    Side effect: each stateful submodule has its ``_module_absolute_name`` set
    so subsequent ``get_state`` calls can find its slot.
    """
    result: ModelState = {}
    for module_name, module in model.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = module_name
            result[module_name] = module.init_state(batch_size, sequence_length)
    return result


def increment_steps(
    model: nn.Module, model_state: ModelState, increment: int = 1
) -> None:
    """Bump the step counter for every stateful submodule of ``model``.

    Uses each module's ``_module_absolute_name`` (set by :func:`init_states`)
    to look up its slot, so this works on subtrees even when ``init_states``
    was called on a different root.
    """
    for _, module in model.named_modules():
        if (
            isinstance(module, StatefulModule)
            and module._module_absolute_name is not None
        ):
            module.increment_step(model_state[module._module_absolute_name], increment)


def reorder_states(
    model: nn.Module, model_state: ModelState, indices: torch.Tensor
) -> None:
    """Reorder every stateful module using a common beam index tensor."""
    for _, module in model.named_modules():
        if (
            isinstance(module, StatefulModule)
            and module._module_absolute_name is not None
        ):
            state = model_state[module._module_absolute_name]
            module.reorder_state(state, indices)


def state_size_bytes(model_state: ModelState) -> int:
    """Return the number of tensor bytes held by an inference state."""

    def visit(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(visit(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(v) for v in value)
        # mamba_ssm's InferenceParams stores per-layer cache tensors in a
        # dictionary.  Avoid importing the optional CUDA package here.
        cache = getattr(value, "key_value_memory_dict", None)
        return visit(cache) if cache is not None else 0

    return visit(model_state)

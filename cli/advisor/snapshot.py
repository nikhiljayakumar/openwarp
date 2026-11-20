"""Immutable, typed readings. Raw API/CLI values are parsed to ints/None here."""
from __future__ import annotations

from dataclasses import dataclass

from cli.agent import AgentError


@dataclass(frozen=True)
class LoadedModel:
    name: str
    size: int
    size_vram: int
    context_length: int | None = None


@dataclass(frozen=True)
class GpuReading:
    index: int | None
    name: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    utilization_percent: int | None = None


@dataclass(frozen=True)
class KvLayout:
    """Per-token KV cache shape of one model: K and V for every layer and KV head."""
    model: str
    layers: int
    kv_heads: int
    key_length: int
    value_length: int


@dataclass(frozen=True)
class MeasuredRequest:
    model: str
    tokens: int  # prompt + output tokens Ollama evaluated; a cached prompt prefix is not counted


@dataclass(frozen=True)
class Snapshot:
    models: tuple[LoadedModel, ...]
    gpus: tuple[GpuReading, ...] = ()  # empty: nvidia-smi unavailable
    kv_layouts: tuple[KvLayout, ...] = ()  # only models /api/show could size
    requests: tuple[MeasuredRequest, ...] = ()  # empty: nothing generated this session


def _int_or_none(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None  # e.g. nvidia-smi's 'N/A' or '[Not Supported]'


def parse_models(payload) -> tuple[LoadedModel, ...]:
    """Parse an Ollama /api/ps JSON body."""
    if not isinstance(payload, dict) or not isinstance(payload.get('models'), list):
        raise AgentError('Ollama /api/ps response must contain a list named "models".')
    models = []
    for item in payload['models']:
        if not isinstance(item, dict) or type(item.get('size')) is not int or type(item.get('size_vram')) is not int:
            raise AgentError('Ollama /api/ps returned a model without integer size and size_vram.')
        context = item.get('context_length')
        models.append(LoadedModel(str(item.get('name', 'unknown')), item['size'], item['size_vram'],
                                  context if type(context) is int else None))
    return tuple(models)


def parse_kv_layout(name: str, payload) -> KvLayout | None:
    """Parse an Ollama /api/show JSON body; None if the model's KV cache can't be sized."""
    info = payload.get('model_info') if isinstance(payload, dict) else None
    if not isinstance(info, dict) or not isinstance(info.get('general.architecture'), str):
        return None
    arch = info['general.architecture'] + '.'
    # Sliding-window and MLA caches don't grow with every layer x head; sizing them this way overstates them.
    if arch + 'attention.sliding_window' in info or arch + 'attention.kv_lora_rank' in info:
        return None
    layers, heads, width = (info.get(arch + k) for k in ('block_count', 'attention.head_count', 'embedding_length'))
    if not all(type(v) is int and v > 0 for v in (layers, heads, width)):
        return None
    head_dim = width // heads  # GGUF default when key/value_length are absent
    shape = (info.get(arch + 'attention.head_count_kv', heads),  # a per-layer list on hybrid models
             info.get(arch + 'attention.key_length', head_dim), info.get(arch + 'attention.value_length', head_dim))
    if not all(type(v) is int and v > 0 for v in shape):
        return None
    return KvLayout(name, layers, *shape)


def parse_gpus(metrics: dict) -> tuple[GpuReading, ...]:
    """Parse a cli.gpu.snapshot() result."""
    return tuple(GpuReading(_int_or_none(g['index']), g['name'],
                            _int_or_none(g['memory_used_mib']), _int_or_none(g['memory_total_mib']),
                            _int_or_none(g.get('utilization_percent')))
                 for g in metrics['gpus'])

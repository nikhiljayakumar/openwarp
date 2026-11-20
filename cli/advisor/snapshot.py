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


@dataclass(frozen=True)
class Snapshot:
    models: tuple[LoadedModel, ...]
    gpus: tuple[GpuReading, ...] = ()  # empty: nvidia-smi unavailable


def _int_or_none(value) -> int | None:
    try:
        return int(float(value))
    except ValueError:
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


def parse_gpus(metrics: dict) -> tuple[GpuReading, ...]:
    """Parse a cli.gpu.snapshot() result."""
    return tuple(GpuReading(_int_or_none(g['index']), g['name'],
                            _int_or_none(g['memory_used_mib']), _int_or_none(g['memory_total_mib']))
                 for g in metrics['gpus'])

"""Flag models that Ollama split between CPU and GPU, and pick the likely cause."""
from __future__ import annotations

from cli.advisor.model import Citation, Confidence, Evidence, Finding, FixOption
from cli.advisor.rule import Rule
from cli.advisor.snapshot import LoadedModel, Snapshot

MIB = 1024 * 1024
LARGE_CONTEXT = 8192
# Windows/WSL desktops idle at 2-3 GB of VRAM; below this, "other apps" is noise.
OTHER_VRAM_LIMIT_MIB = 3072
# Ollama (llama.cpp) runs offloaded layers on the CPU from system RAM; the weights
# stay put, but every token waits on slow CPU layers and crosses PCIe between them.
WHY_SLOW = ('A partial CPU offload is slow: every token must pass through the layers left in '
            'system RAM, which the CPU computes with far less memory bandwidth than the GPU, '
            'and activations cross PCIe between CPU and GPU layers.')

# Each measured number appears once, in the evidence rows; cause, fixes and
# confidence basis refer to it without repeating it.
Cause = tuple[str, tuple[FixOption, ...], Confidence]


def _gb(value: int) -> str:
    return f'{value / 1e9:.1f} GB'


def _systemd(*env: str) -> tuple[str, ...]:
    return ('sudo systemctl edit ollama', 'add under [Service]:',
            *(f'  Environment="{e}"' for e in env), 'sudo systemctl restart ollama')


def _other_vram_mib(snap: Snapshot) -> int | None:
    """VRAM in use that Ollama's loaded models don't account for, or None if unknown."""
    used = [g.memory_used_mib for g in snap.gpus]
    if not used or None in used:
        return None
    return max(0, round(sum(used) - sum(m.size_vram for m in snap.models) / MIB))


def _multiple_models(model: LoadedModel, snap: Snapshot) -> Cause | None:
    others = [m.name for m in snap.models if m is not model]
    if not others:
        return None
    return ('Several loaded models competing for VRAM',
            (FixOption('Unload idle models', tuple(f'ollama stop {name}' for name in others)),
             FixOption('Keep one resident', _systemd('OLLAMA_MAX_LOADED_MODELS=1'))),
            Confidence('high', 'Measured: /api/ps lists more than one loaded model.'))


def _kv_cache(model: LoadedModel) -> Cause | None:
    if model.context_length is None or model.context_length <= LARGE_CONTEXT:
        return None
    return ('KV cache for the long context',
            (FixOption('Shorter context', (f'/set parameter num_ctx {LARGE_CONTEXT}',), 'inside ollama run'),
             FixOption('Quantize KV cache', _systemd('OLLAMA_FLASH_ATTENTION=1', 'OLLAMA_KV_CACHE_TYPE=q8_0'))),
            Confidence('high', 'Measured: the context is above the threshold while part of the model is on CPU.'))


def _other_apps(other_mib: int | None) -> Cause | None:
    if other_mib is None or other_mib <= OTHER_VRAM_LIMIT_MIB:
        return None
    return ('Other GPU apps holding VRAM',
            (FixOption('Close other GPU apps', ('nvidia-smi',),
                       'browsers, games, other inference servers; a Windows/WSL desktop '
                       'holds 2-3 GB that cannot be reclaimed'),),
            Confidence('medium', 'Inferred: nvidia-smi usage minus Ollama size_vram is over the threshold; '
                                 'the processes using it are not identified.'))


def _weights_too_large(model: LoadedModel) -> Cause:
    return ('Weights larger than free VRAM',
            (FixOption('Smaller quantization', (f'ollama show {model.name}',), 'e.g. Q4_K_M'),
             FixOption('Smaller model', ('ollama list',))),
            Confidence('low', 'Fallback: no measured cause matched, so the weights alone are assumed not to fit.'))


def _evidence(model: LoadedModel, snap: Snapshot, other_mib: int | None) -> tuple[Evidence, ...]:
    cpu_bytes = model.size - model.size_vram
    cpu_pct = round(cpu_bytes / model.size * 100)
    context = model.context_length
    return (
        Evidence('Placement', f'{cpu_pct}%/{100 - cpu_pct}% CPU/GPU', f'{_gb(cpu_bytes)} of {_gb(model.size)} on CPU'),
        Evidence('Context', 'unknown') if context is None else
        Evidence('Context', f'{context:,} tokens',
                 f'above the {LARGE_CONTEXT:,} threshold' if context > LARGE_CONTEXT else None),
        Evidence('Loaded models', str(len(snap.models))),
        Evidence('Other VRAM', 'unknown', 'GPU snapshot unavailable') if other_mib is None else
        Evidence('Other VRAM', f'{other_mib:,} MiB',
                 f'above the {OTHER_VRAM_LIMIT_MIB:,} MiB threshold' if other_mib > OTHER_VRAM_LIMIT_MIB else None),
    )


class CpuOffloadRule(Rule):
    id = 'cpu-offload'
    citation = Citation('Song et al.', 'PowerInfer: Fast LLM Serving with a Consumer-grade GPU',
                        'SOSP', 2024, '2312.12456')
    passed_message = 'No offload detected: all loaded models are 100% on GPU.'

    def evaluate(self, snap: Snapshot) -> list[Finding]:
        other_mib = _other_vram_mib(snap)
        findings = []
        for model in snap.models:
            if model.size <= 0 or model.size_vram >= model.size:
                continue
            # Directly measured causes first, then inferred, then the fallback.
            cause, fixes, confidence = (_multiple_models(model, snap) or _kv_cache(model)
                                        or _other_apps(other_mib) or _weights_too_large(model))
            findings.append(Finding(
                self.id, 'CPU offload', model.name,
                'Part of the model is running on the CPU, slowing every token.',
                _evidence(model, snap, other_mib), cause, fixes, WHY_SLOW, confidence, self.citation))
        return findings

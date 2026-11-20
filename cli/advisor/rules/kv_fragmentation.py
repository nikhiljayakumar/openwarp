"""Flag KV caches reserved far beyond the tokens that use them, and suggest PagedAttention."""
from __future__ import annotations

from cli.advisor.model import Citation, Confidence, Evidence, Finding, FixOption
from cli.advisor.rule import Rule
from cli.advisor.snapshot import KvLayout, LoadedModel, Snapshot

MIB = 1024 * 1024
F16_BYTES = 2
RESERVED_LIMIT_MIB = 1024
MIN_CONTEXT = 2048  

WHY_WASTED = ('Ollama reserves the KV cache for the full context window when the model loads, so '
              'VRAM held for tokens a request never reaches is wasted. PagedAttention allocates the '
              'cache in small blocks as tokens arrive, so only the last block of a request is partly empty.')

Cause = tuple[str, tuple[FixOption, ...], Confidence]


def _bytes_per_token(layout: KvLayout) -> int:
    return layout.layers * layout.kv_heads * (layout.key_length + layout.value_length) * F16_BYTES


def _sized(snap: Snapshot) -> list[tuple[LoadedModel, KvLayout]]:
    """Loaded models whose context length and attention layout are both known."""
    layouts = {layout.model: layout for layout in snap.kv_layouts}
    return [(m, layouts[m.name]) for m in snap.models if m.context_length and m.name in layouts]


def _tokens_used(model: LoadedModel, snap: Snapshot) -> int | None:
    # A request sent as 'qwen2.5' shows up in /api/ps as 'qwen2.5:latest'.
    names = {model.name, model.name.removesuffix(':latest')}
    tokens = [r.tokens for r in snap.requests if r.model in names]
    return min(tokens[-1], model.context_length) if tokens else None


def _paged_attention(model: LoadedModel) -> FixOption:
    return FixOption('PagedAttention (vLLM)', ('pip install vllm', 'vllm serve HF_MODEL'),
                     f'in a venv; HF_MODEL is the Hugging Face repo of {model.name}')


def _short_requests(model: LoadedModel, used: int) -> Cause:
    fit = max(MIN_CONTEXT, 1 << (used - 1).bit_length())
    fixes = (_paged_attention(model),)
    if fit < model.context_length:  # old full-attention models can pass the limit at 2,048 tokens
        fixes += (FixOption('Smaller context', (f'/set parameter num_ctx {fit}',), 'inside ollama run'),)
    return ('Context window far larger than the last request', fixes,
            Confidence('high', "Measured: Ollama's token count for the last request against the reserved window."))


def _unmeasured(model: LoadedModel) -> Cause:
    return ('Context window likely larger than the requests',
            (_paged_attention(model),
             FixOption('Measure a request', ('/perf',), 'in openwarp, then /advise again')),
            Confidence('low', 'Fallback: no request was measured this session, so short requests are assumed.'))


def _evidence(model: LoadedModel, layout: KvLayout, used: int | None) -> tuple[Evidence, ...]:
    context, per_token = model.context_length, _bytes_per_token(layout)
    return (
        Evidence('Context', f'{context:,} tokens'),
        Evidence('Per token', f'{per_token / 1024:,.0f} KiB',
                 f'f16 K and V, {layout.layers} layers × {layout.kv_heads} KV heads'),
        Evidence('KV reserved', f'{context * per_token / MIB:,.0f} MiB',
                 f'above the {RESERVED_LIMIT_MIB:,} MiB threshold'),
        Evidence('Last request', 'not measured') if used is None else
        Evidence('Last request', f'{used:,} tokens', f'{used / context:.1%} of the context'),
    )


class KvFragmentationRule(Rule):
    id = 'kv-fragmentation'
    citation = Citation('Kwon et al.', 'Efficient Memory Management for Large Language Model Serving '
                        'with PagedAttention', 'SOSP', 2023, '2309.06180')
    passed_message = (f'No KV cache fragmentation: each KV cache is under {RESERVED_LIMIT_MIB:,} MiB '
                      'or at least half in use.')

    def applies(self, snap: Snapshot) -> bool:
        return bool(_sized(snap))

    def skip_reason(self, snap: Snapshot) -> str:
        if not snap.models:
            return super().skip_reason(snap)
        return 'Ollama did not report a context length and a supported attention layout for any loaded model.'

    def evaluate(self, snap: Snapshot) -> list[Finding]:
        findings = []
        for model, layout in _sized(snap):
            used = _tokens_used(model, snap)
            if model.context_length * _bytes_per_token(layout) <= RESERVED_LIMIT_MIB * MIB:
                continue
            if used is not None and used * 2 > model.context_length:
                continue  # at least half in use: the window fits the work
            # A measured request decides the cause; without one, short requests are assumed.
            cause, fixes, confidence = _unmeasured(model) if used is None else _short_requests(model, used)
            findings.append(Finding(
                self.id, 'KV cache fragmentation', model.name,
                'Most of the reserved KV cache holds no tokens, wasting VRAM.',
                _evidence(model, layout, used), cause, fixes, WHY_WASTED, confidence, self.citation))
        return findings

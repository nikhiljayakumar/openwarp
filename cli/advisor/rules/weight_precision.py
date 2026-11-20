"""Flag models fully on the GPU that run 8- or 16-bit weights, and suggest a 4- or 5-bit build."""
from __future__ import annotations

from cli.advisor.model import Citation, Confidence, Evidence, Finding, FixOption
from cli.advisor.rule import Rule
from cli.advisor.snapshot import Snapshot

TARGET, TARGET_BITS = 'Q4_K_M', 4.89
ACCURATE = 'Q5_K_M'
TAGS = {TARGET: 'q4_K_M', ACCURATE: 'q5_K_M'}  # how Ollama spells them in model tags
# Bits per weight for the levels that trigger (llama.cpp GGUF types).
BITS = {'F32': 32.0, 'F16': 16.0, 'BF16': 16.0, 'Q8_0': 8.5}
TAG_SUFFIXES = ('q8_0', 'fp16', 'f16', 'bf16', 'fp32', 'f32')
# ponytail: size floor skips embedding models and tiny chat models; use /api/show capabilities if it misfires.
MIN_SIZE = 2_000_000_000
WHY = ('Generating each token reads the weights it uses from memory, so with one request at a time the '
       'speed is set by memory bandwidth, not compute. Fewer bits per weight means fewer bytes per token. '
       '4-bit costs a little accuracy, but across model families it almost always gives the best accuracy '
       'for a given memory size; 5-bit is in between on both speed and accuracy.')


def _swap_suffix(name: str, level: str) -> str | None:
    """Return name with its quantization tag suffix replaced by level's tag, or None if it has none."""
    tag = name.rpartition(':')[2].lower()
    for suffix in TAG_SUFFIXES:
        if tag.endswith('-' + suffix):
            return name[:-len(suffix)] + TAGS[level]
    return None


def _fixes(name: str) -> tuple[FixOption, ...]:
    fast, accurate = _swap_suffix(name, TARGET), _swap_suffix(name, ACCURATE)
    if fast is None:
        return (FixOption(f'Find a {TARGET} or {ACCURATE} build', (f'ollama show {name}',),
                          f'pick a {TAGS[TARGET]} or {TAGS[ACCURATE]} tag from its model page'),)
    return (FixOption(f'Switch to {TARGET}', (f'ollama pull {fast}',), 'fastest; then select it with /ollama'),
            FixOption(f'Switch to {ACCURATE}', (f'ollama pull {accurate}',),
                      'more accurate, slower; not every model has it'))


class WeightPrecisionRule(Rule):
    id = 'weight-precision'
    citation = Citation('Dettmers & Zettlemoyer', 'The case for 4-bit precision: k-bit Inference Scaling Laws',
                        'ICML', 2023, '2212.09720')
    passed_message = 'Weights OK: no model fully on the GPU uses 8- or 16-bit weights.'

    def evaluate(self, snap: Snapshot) -> list[Finding]:
        findings = []
        for model in snap.models:
            level = (model.quantization or '').upper()
            bits = BITS.get(level)
            if bits is None or model.size < MIN_SIZE or model.size_vram < model.size:
                continue  # low/unknown precision, too small (e.g. embeddings), or partly on CPU (cpu-offload's job)
            findings.append(Finding(
                self.id, 'Weight precision', model.name,
                'Each token reads more weight bytes than a 4-bit build would.',
                (Evidence('Weights', level, f'{bits:g} bits per weight'), Evidence('Placement', '100% GPU')),
                f'Weights read per token: about {bits / TARGET_BITS:.1f}x a {TARGET} build',
                _fixes(model.name), WHY,
                Confidence('medium', 'Inferred: the quantization level is read from /api/ps; '
                                     'the speedup is estimated from bytes per token, not measured.'),
                self.citation))
        return findings

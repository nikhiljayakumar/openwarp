"""Performance reported by Ollama for a real generation request."""
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median

from cli.agent import OllamaAgent


def _number(result: dict, key: str) -> int | float | None:
    value = result.get(key)
    return value if type(value) in (float, int) and math.isfinite(value) and value >= 0 else None


def _rate(result: dict, phase: str) -> float | None:
    count = _number(result, f'{phase}_count')
    duration = _number(result, f'{phase}_duration')
    return count / duration * 1e9 if count is not None and duration else None


BASELINE_MIN_SAMPLES = 5
BASELINE_WINDOW = 12
SLOW_RATE_FRACTION = 0.75
SLOW_RUNS_TO_ALERT = 3
MAX_WARM_LOAD_NS = 1_000_000_000


@dataclass
class _RateHistory:
    rates: deque[float] = field(default_factory=lambda: deque(maxlen=BASELINE_WINDOW))
    consecutive_slow: int = 0


@dataclass(frozen=True)
class Slowdown:
    phase: str
    rate: float
    baseline: float

    def __str__(self) -> str:
        return (f'{self.phase} slowdown: {self.rate:.2f} tokens/s vs '
                f'{self.baseline:.2f} baseline for comparable requests '
                f'({SLOW_RUNS_TO_ALERT} slow samples in a row).')


class PerformanceBaseline:
    """Track comparable Ollama requests within one interactive session."""

    def __init__(self) -> None:
        self._histories: dict[tuple[str, str, int, str, int], _RateHistory] = {}

    def observe(self, agent: OllamaAgent, result: dict) -> list[Slowdown]:
        load = _number(result, 'load_duration')
        prompt_count = _number(result, 'prompt_eval_count')
        if load is None or load >= MAX_WARM_LOAD_NS or not prompt_count:
            return []

        # Group request sizes by powers of two, such as 32–63 and 64–127 tokens.
        prompt_bucket = int(prompt_count).bit_length()
        alerts = []
        for phase, label in (('prompt_eval', 'Prompt processing'), ('eval', 'Generation')):
            count = _number(result, f'{phase}_count')
            rate = _rate(result, phase)
            if not count or not rate:
                continue
            key = (agent.endpoint, agent.model, prompt_bucket, phase, int(count).bit_length())
            history = self._histories.setdefault(key, _RateHistory())
            reference = median(history.rates) if len(history.rates) >= BASELINE_MIN_SAMPLES else None
            if reference is not None and rate <= reference * SLOW_RATE_FRACTION:
                history.consecutive_slow += 1
                if history.consecutive_slow == SLOW_RUNS_TO_ALERT:
                    alerts.append(Slowdown(label, rate, reference))
                continue
            history.consecutive_slow = 0
            history.rates.append(rate)
        return alerts


def format_performance(result: dict) -> str:
    def seconds(key):
        value = _number(result, key)
        return f'{value / 1e9:.3f} s' if value is not None else 'N/A'

    def count(key):
        value = _number(result, key)
        return str(value) if value is not None else 'N/A'

    def rate(phase):
        value = _rate(result, phase)
        return f'{value:.2f} tokens/s' if value is not None else 'N/A'

    return '\n'.join([
        'Ollama performance · measured generation',
        f"  Model           {result.get('model', 'unknown')}",
        f"  Total time      {seconds('total_duration')}",
        f"  Load time       {seconds('load_duration')}",
        f"  Prompt tokens   {count('prompt_eval_count')}",
        f"  Prompt time     {seconds('prompt_eval_duration')}",
        f"  Prompt rate     {rate('prompt_eval')}",
        f"  Output tokens   {count('eval_count')}",
        f"  Generate time   {seconds('eval_duration')}",
        f"  Generate rate   {rate('eval')}",
        '',
        'Source: Ollama API · one request; CPU/GPU placement: ollama ps',
    ])


DEFAULT_PROMPT = 'Explain in a short paragraph how a computer processes a user request.'
PERF_OPTIONS = {'num_predict': 128, 'temperature': 0}


def report_performance(agent: OllamaAgent, prompt: str | None = None,
                       baseline: PerformanceBaseline | None = None, *,
                       on_slowdown: Callable[[OllamaAgent, str], None] | None = None) -> None:
    print(f'Running a short generation with {agent.model}...', flush=True)
    prompt = DEFAULT_PROMPT if prompt is None else prompt
    result = agent.generate(prompt, options=PERF_OPTIONS)
    print('\nModel response (up to 128 output tokens):')
    print(result['response'])
    print()
    print(format_performance(result))
    if baseline is not None:
        alerts = baseline.observe(agent, result)
        for alert in alerts:
            print(f'  {alert}')
        if alerts and on_slowdown is not None:
            on_slowdown(agent, prompt)

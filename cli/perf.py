"""Performance reported by Ollama for a real generation request."""
import math

from cli.agent import OllamaAgent


def format_performance(result: dict) -> str:
    def number(key):
        value = result.get(key)
        return value if type(value) in (float, int) and math.isfinite(value) and value >= 0 else None

    def seconds(key):
        value = number(key)
        return f'{value / 1e9:.3f} s' if value is not None else 'N/A'

    def count(key):
        value = number(key)
        return str(value) if value is not None else 'N/A'

    def rate(count_key, duration_key):
        tokens, duration = number(count_key), number(duration_key)
        if tokens is None or duration is None or duration == 0:
            return 'N/A'
        return f'{tokens / duration * 1e9:.2f} tokens/s'

    return '\n'.join([
        'Ollama performance · measured generation',
        f"  Model           {result.get('model', 'unknown')}",
        f"  Total time      {seconds('total_duration')}",
        f"  Load time       {seconds('load_duration')}",
        f"  Prompt tokens   {count('prompt_eval_count')}",
        f"  Prompt time     {seconds('prompt_eval_duration')}",
        f"  Prompt rate     {rate('prompt_eval_count', 'prompt_eval_duration')}",
        f"  Output tokens   {count('eval_count')}",
        f"  Generate time   {seconds('eval_duration')}",
        f"  Generate rate   {rate('eval_count', 'eval_duration')}",
        '',
        'Source: Ollama API · one request; CPU/GPU placement: ollama ps',
    ])


DEFAULT_PROMPT = 'Explain in a short paragraph how a computer processes a user request.'


def report_performance(agent: OllamaAgent, prompt: str | None = None) -> None:
    print(f'Running a short generation with {agent.model}...', flush=True)
    result = agent.generate(
        prompt if prompt is not None else DEFAULT_PROMPT,
        options={'num_predict': 128, 'temperature': 0},
    )
    print('\nModel response (up to 128 output tokens):')
    print(result['response'])
    print()
    print(format_performance(result))

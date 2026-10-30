"""Read-only NVIDIA GPU snapshots; no tuning or diagnosis engine."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import subprocess

from cli.agent import Agent, AgentError

FIELDS = ('index', 'name', 'utilization.gpu', 'memory.used', 'memory.total',
          'temperature.gpu', 'power.draw')
KEYS = ('index', 'name', 'utilization_percent', 'memory_used_mib',
        'memory_total_mib', 'temperature_c', 'power_w')


def snapshot() -> dict:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=' + ','.join(FIELDS), '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except FileNotFoundError as exc:
        raise AgentError('nvidia-smi is unavailable. GPU snapshots currently require an NVIDIA GPU and driver.') from exc
    except (subprocess.SubprocessError, OSError) as exc:
        raise AgentError('Could not read GPU metrics from nvidia-smi. Check your NVIDIA driver and GPU access.') from exc
    gpus = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        if len(row) != len(KEYS):
            raise AgentError('Unexpected GPU metrics format from nvidia-smi.')
        # Keep unavailable sensor readings explicit rather than inventing numbers.
        gpus.append(dict(zip(KEYS, (value.strip() for value in row))))
    if not gpus:
        raise AgentError('nvidia-smi returned no GPUs.')
    return {'captured_at': datetime.now(timezone.utc).isoformat(), 'source': 'nvidia-smi', 'gpus': gpus}


def format_snapshot(metrics: dict) -> str:
    """Readable GPU cards, also suitable for piped output."""
    lines = ['GPU performance', f"Captured: {metrics.get('captured_at', 'unknown')}  |  Source: {metrics.get('source', 'unknown')}"]

    def value(gpu: dict, key: str, unit: str) -> str:
        reading = gpu.get(key, 'N/A')
        if reading in {'N/A', '[N/A]', '[Not Supported]', 'Not Supported', ''}:
            return 'N/A'
        return f'{reading}{unit}'

    for gpu in metrics['gpus']:
        lines.extend([
            '',
            f"GPU {gpu.get('index', '?')} · {gpu.get('name', 'Unknown GPU')}",
            f"  Utilization   {value(gpu, 'utilization_percent', '%')}",
            f"  VRAM          {value(gpu, 'memory_used_mib', ' MiB')} / {value(gpu, 'memory_total_mib', ' MiB')}",
            f"  Temperature   {value(gpu, 'temperature_c', ' °C')}",
            f"  Power         {value(gpu, 'power_w', ' W')}",
        ])
    return '\n'.join(lines)


def report_gpu(agent: Agent | None = None, explain: bool = False) -> None:
    if explain and agent is None:
        raise AgentError('GPU explanation requires --ollama-model MODEL or an agent connection.')
    metrics = snapshot()
    card = format_snapshot(metrics)
    print(card, flush=True)
    if explain:
        print('\nModel explanation (not a measured diagnosis):', flush=True)
        print(agent.reply(
            'Here are the measured GPU readings:\n\n'
            + card
            + '\n\nExplain these readings in 3 short bullet points, citing their numbers. '
            'Describe current GPU activity and memory use. '
            'N/A means a sensor reading is unavailable. '
            'This is one snapshot taken before your reply, not a benchmark or a trend. '
            'Do not invent readings or claim a proven bottleneck.'
        ))

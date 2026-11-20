"""Collect diagnostics after an interactive /perf slowdown."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
from urllib.parse import urlsplit

from cli.advisor.collect import collect
from cli.advisor.engine import RuleEngine
from cli.advisor.render import render_text
from cli.advisor.rules import DEFAULT_RULES
from cli.agent import AgentError, OllamaAgent


DEFAULT_PROFILE_OUTPUT = Path('profiles')


def _run(command: list[str], log: Path, timeout: int = 180) -> None:
    # The profiler, replay worker, and Ollama runner must all stop on cancellation.
    python_paths = [str(Path(__file__).resolve().parents[1])]
    if os.environ.get('PYTHONPATH'):
        python_paths.append(os.environ['PYTHONPATH'])
    with log.open('w', encoding='utf-8') as output:
        with subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                              env={**os.environ, 'PYTHONPATH': os.pathsep.join(python_paths)},
                              start_new_session=True) as process:
            try:
                status = process.wait(timeout=timeout)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise
    if status:
        raise AgentError(f'{command[0]} exited with status {status}. See {log}.')


def _top_kernel(summary: Path) -> str | None:
    with summary.open(newline='', encoding='utf-8') as source:
        for line in source:
            # nsys prints progress messages before the CSV table.
            header = next(csv.reader([line]), [])
            if 'Total Time (ns)' in header:
                rows = (row for row in csv.DictReader(source, fieldnames=header)
                        if row.get('Name') and (row.get('Total Time (ns)') or '').strip().isdigit())
                top = max(rows, key=lambda row: int(row['Total Time (ns)']), default=None)
                return top['Name'] if top else None
    return None


def _capture(agent: OllamaAgent, prompt: str, output: Path) -> None:
    if not shutil.which('nsys') or not shutil.which('ollama'):
        raise AgentError('Install Nsight Systems (nsys) and Ollama on the GPU host.')
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix='slowdown-', dir=output.resolve()))
    print(f'Capturing a fresh Ollama run. Reports: {directory}', flush=True)
    trace = directory / 'timeline.nsys-rep'
    summary = directory / 'kernels.csv'
    with TemporaryDirectory() as temporary:
        request = Path(temporary) / 'request.json'
        request.write_text(json.dumps({'model': agent.model, 'prompt': prompt}), encoding='utf-8')
        worker = [sys.executable, '-m', 'cli.profile_workload', str(request)]
        _run(['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
              '--output', str(trace.with_suffix('')), *worker], directory / 'timeline.log')
        _run(['nsys', 'stats', '--report', 'cuda_gpu_kern_sum:mangled',
              '--format', 'csv', str(trace)], summary, timeout=30)
        kernel = _top_kernel(summary)
        if not kernel:
            print('Nsight Systems reported no CUDA kernel timings.')
            return
        print(f'Kernel with the most aggregate GPU time: {kernel}')
        if not shutil.which('ncu'):
            print('Install Nsight Compute (ncu) to collect kernel counters.')
            return
        counters = directory / 'kernel.ncu-rep'
        _run(['ncu', '--target-processes', 'all', '--kernel-name-base', 'mangled',
              '--kernel-name', f'regex:^{re.escape(kernel)}$', '--launch-count', '3',
              '--section', 'SpeedOfLight', '--section', 'MemoryWorkloadAnalysis',
              '--section', 'ComputeWorkloadAnalysis', '--section', 'Occupancy',
              '--section', 'SchedulerStats', '--export', str(counters.with_suffix('')),
              *worker], directory / 'compute.log')
        if not counters.is_file():
            raise AgentError(f'No counter report was produced. See {directory / "compute.log"}.')
        _run(['ncu', '--import', str(counters), '--page', 'details'],
             directory / 'kernel-analysis.txt', timeout=30)
        print(f'Kernel counters and Nsight analysis: {directory / "kernel-analysis.txt"}')


def investigate_slowdown(agent: OllamaAgent, prompt: str, output: Path = DEFAULT_PROFILE_OUTPUT) -> None:
    base_url = agent.endpoint.removesuffix('/api/generate')
    local = urlsplit(base_url).hostname in {'localhost', '127.0.0.1', '::1'}
    try:
        snapshot = collect(base_url, include_local_gpu=local)
        for gpu in snapshot.gpus:
            utilization = 'N/A' if gpu.utilization_percent is None else f'{gpu.utilization_percent}%'
            print(f'Post-request GPU snapshot · {gpu.name}: {utilization} utilization, '
                  f'{gpu.memory_used_mib if gpu.memory_used_mib is not None else "N/A"}/'
                  f'{gpu.memory_total_mib if gpu.memory_total_mib is not None else "N/A"} MiB VRAM')
        report = RuleEngine(DEFAULT_RULES).run(snapshot)
        print(render_text(report), flush=True)
    except AgentError as exc:
        print(f'Slowdown diagnostics unavailable: {exc}')
        return
    if not local:
        print('Run Nsight capture on the Ollama GPU host.')
        return
    model = agent.model if ':' in agent.model.rsplit('/', 1)[-1] else f'{agent.model}:latest'
    placement = next((item for item in snapshot.models if item.name in {agent.model, model}), None)
    if placement is None or placement.size <= 0:
        print('Skipping kernel capture because model placement is unavailable.')
        return
    if placement.size_vram < placement.size:
        print('Skipping kernel capture because the model is running partly or entirely on CPU.')
        return
    try:
        _capture(agent, prompt, output)
    except (AgentError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'Kernel capture incomplete: {exc}')

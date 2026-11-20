"""The only advisor module that performs I/O."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from cli import gpu
from cli.agent import AgentError
from cli.advisor.snapshot import LoadedModel, Snapshot, parse_gpus, parse_models


def ollama_placement(base_url: str) -> tuple[LoadedModel, ...]:
    if urlsplit(base_url).scheme not in {'http', 'https'} or not urlsplit(base_url).netloc:
        raise AgentError('Ollama URL must be an absolute HTTP or HTTPS URL.')
    endpoint = base_url.rstrip('/') + '/api/ps'
    try:
        with urlopen(endpoint, timeout=10) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise AgentError(f'Ollama returned HTTP {exc.code} for /api/ps. Check journalctl -u ollama -n 50 --no-pager.') from exc
    except TimeoutError as exc:
        raise AgentError('Ollama timed out on /api/ps. Check ollama ps and the Ollama service logs.') from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise AgentError('Ollama timed out on /api/ps. Check ollama ps and the Ollama service logs.') from exc
        raise AgentError(f'Cannot connect to {endpoint}. Check the URL and Ollama service.') from exc
    except ValueError as exc:
        raise AgentError('Ollama returned invalid JSON.') from exc
    except OSError as exc:
        raise AgentError('Ollama connection interrupted. Check service logs.') from exc
    return parse_models(payload)


def collect(base_url: str, *, include_local_gpu: bool = True) -> Snapshot:
    models = ollama_placement(base_url)
    if not include_local_gpu:
        return Snapshot(models)
    try:
        gpus = parse_gpus(gpu.snapshot())
    except AgentError:
        gpus = ()  # no nvidia-smi: rules that need GPU readings skip those checks
    return Snapshot(models, gpus)

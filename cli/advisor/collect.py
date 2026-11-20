"""The only advisor module that performs I/O."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from cli import gpu
from cli.agent import AgentError
from cli.advisor.snapshot import (KvLayout, LoadedModel, MeasuredRequest, Snapshot, parse_gpus,
                                  parse_kv_layout, parse_models)


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


def ollama_kv_layout(base_url: str, name: str) -> KvLayout | None:
    request = Request(base_url.rstrip('/') + '/api/show', data=json.dumps({'model': name}).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (OSError, ValueError):
        return None  # /api/ps already answered; a model without a layout is skipped, not an error
    return parse_kv_layout(name, payload)


def collect(base_url: str, requests: tuple[MeasuredRequest, ...] = (), *,
            include_local_gpu: bool = True) -> Snapshot:
    models = ollama_placement(base_url)
    gpus = ()  # local nvidia-smi can't see a remote Ollama server's GPUs
    if include_local_gpu:
        try:
            gpus = parse_gpus(gpu.snapshot())
        except AgentError:
            gpus = ()  # no nvidia-smi: rules that need GPU readings skip those checks
    layouts = tuple(layout for m in models if (layout := ollama_kv_layout(base_url, m.name)))
    return Snapshot(models, gpus, layouts, requests)

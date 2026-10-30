# OpenWarp CLI

A standalone terminal shell with an optional agent API connection. Python 3.11+;
no runtime dependencies.

```sh
python3 -m cli --help
python3 -m cli                 # interactive shell
```

Or install with `python3 -m pip install -e .` and run `openwarp`.
Inside the shell, use `/read` for a GPU snapshot, `/help`, `/exit`, or `/quit`. Ctrl-D or Ctrl-C exits.

## Optional agent connection

```sh
openwarp --agent-url http://localhost:8000/chat
openwarp --agent-url http://localhost:8000/chat "Hello"
```

You can also set `OPENWARP_AGENT_URL` and an optional `OPENWARP_AGENT_TOKEN`
(bearer token). Each message sends an HTTP POST with JSON `{"message": "Hello"}`;
the endpoint must return JSON `{"reply": "Hi"}`. Requests time out after 30 seconds.
This is a small generic adapter contract, not a provider-specific API or a bundled
agent. Conversation state is the endpoint's responsibility. The CLI does not
persist messages or credentials. Without an endpoint, the shell runs locally
and explains how to connect when you enter a message.

The `Agent` protocol in `cli/agent.py` is the extension point for another API.

Run checks with `python3 -m unittest discover -s tests -v`.

## GPU snapshots explained by Ollama

OpenWarp reads sensors using NVIDIA's `nvidia-smi`; Ollama receives those readings
as text and explains them. No rules engine, inference engine, GPU tuning, or
additional Python packages are bundled.

Start your installed Ollama server (`ollama serve` if it is not already running),
then use a model shown by `ollama list`. Replace `YOUR_MODEL` below with its name;
install a model using `ollama pull MODEL` if needed.

```sh
# Read metrics without any model:
python3 -m cli --read

# Read metrics and ask your installed Ollama model to explain them:
python3 -m cli --read --explain --ollama-model YOUR_MODEL

# Interactive chat, with /read and /read explain available:
python3 -m cli --ollama-model YOUR_MODEL
```

The default Ollama URL is `http://localhost:11434`; override with `--ollama-url`.
The native `/api/generate` API is used with streaming disabled and a 120-second
request timeout. Ordinary chat does not automatically attach GPU readings.

Snapshots include all GPUs visible to `nvidia-smi`, with utilization (%), used
and total VRAM (MiB), temperature (C), and power (W). Unavailable sensor values
remain explicit. Currently this requires an NVIDIA GPU and a working driver;
AMD and Apple GPU metrics are not supported.

Metrics describe the machine running OpenWarp, even if Ollama is remote. When a model is connected, `/read` and `--read` send readings to it automatically.
Readings are taken before the explanation request, so they do not measure
that request's inference performance. It includes other GPU workloads and cannot
establish a bottleneck or tokens/sec from one reading. Model explanations are
labelled separately from the actual measurements.

API references: [Ollama generate](https://docs.ollama.com/api/generate) and
[NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/).

`--gpu` and `/gpu` remain aliases for `--read` and `/read`. GPU reads work
without an agent connection; explanations are optional.

## Readings with Ollama

With Ollama running, start `python3 -m cli` and type:

```text
/ollama YOUR_MODEL
/read
```

Choose an installed model from `ollama list`. `/read` includes a model explanation
when connected. No model is
automatically downloaded. Without a connection, reads show measurements only.

Readings print as GPU cards with labelled utilization, VRAM, temperature, and
power values, followed by the optional model explanation. Unavailable sensors
show `N/A`. Each read takes a single snapshot.

Set `OPENWARP_OLLAMA_MODEL` and optionally `OPENWARP_OLLAMA_URL` to avoid repeating
connection flags. Analysis uses the observations supplied; it cannot expose
counters the GPU driver does not report or prove an application bottleneck.
It performs no configuration changes.

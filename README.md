# OpenWarp

A terminal CLI for local model chat, Ollama inference measurements, and NVIDIA
GPU readings. Python 3.11+; no third-party Python runtime dependencies.

## Environment

The setup below targets **Ubuntu Linux or Ubuntu in WSL2**, using Bash.
Install and run Ollama in the same environment as OpenWarp. The default server
is `http://localhost:11434`.

- **Chat and `/perf`:** running Ollama server and a downloaded model. CPU or GPU inference works.
- **`/read`:** NVIDIA GPU, working driver, and `nvidia-smi` accessible inside Ubuntu/WSL.
- **Installation:** internet access and sudo. Local inference works offline after downloading a model.

For GPU compatibility, see [Ollama hardware support](https://docs.ollama.com/gpu).
Check `nvidia-smi` for sensor access and `ollama ps` during inference for CPU/GPU
placement. GPU detection during installation alone does not confirm GPU inference.

## Setup

### 1. Prerequisites and clone

```bash
sudo apt-get update
sudo apt-get install -y git python3 curl zstd
python3 --version   # must be 3.11+
git clone REPOSITORY_URL openwarp
cd openwarp
```

Replace `REPOSITORY_URL` with this repository's clone URL. Already cloned? Start
from its root directory. `zstd` is required by the Ollama installer.

### 2. Add the `openwarp` command

```bash
chmod +x bin/openwarp
mkdir -p "$HOME/.local/bin"
ln -s "$PWD/bin/openwarp" "$HOME/.local/bin/openwarp"
export PATH="$HOME/.local/bin:$PATH"
openwarp --version
```

Add the `export PATH=...` line to `~/.bashrc` if absent. If the link already
exists, check it with `ls -l ~/.local/bin/openwarp`. Keep the checkout in place;
the launcher uses its files. Alternatively, run `python3 -m cli` from the checkout.
No pip or virtual environment is required.

### 3. Install Ollama and a starter model

Use the [official Linux installer](https://docs.ollama.com/linux):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
curl -fsS http://localhost:11434/api/tags
```

If the API is unreachable, run `sudo systemctl start ollama`. Without systemd,
run `ollama serve` in another terminal and leave it open. Only one server is needed.

Download [Qwen 2.5 0.5B](https://ollama.com/library/qwen2.5:0.5b) (~398 MB,
plus working RAM/VRAM) and verify it generates a reply:

```bash
ollama pull qwen2.5:0.5b
ollama run qwen2.5:0.5b "Say hello."
```

### 4. Launch

```bash
openwarp
```

Inside OpenWarp:

```text
/ollama
/perf Explain how a GPU works in three sentences.
/read
```

## Commands

| Input | Action |
| --- | --- |
| `/ollama [MODEL]` | Select a model for this session; defaults to `qwen2.5:0.5b`. Does not download it. |
| `/models` | List installed models. |
| `/pull [MODEL]` | Download and select a model; same default as `/ollama`. |
| `/perf [PROMPT]` | Generate a reply and display its inference measurements; uses a built-in prompt if omitted. |
| `/read` | Capture GPU readings; a connected model also explains them. |
| Ordinary text | Chat without collecting GPU readings or displaying performance. |
| `/help` | Show commands. |
| `/exit`, `/quit` | Exit. Ctrl-D or Ctrl-C also exits at the prompt. |

`/gpu` aliases `/read`; `/read explain` explicitly requires a model explanation.

## What gets measured?

**`/perf` runs your prompt through Ollama.** After generation finishes, it prints
the reply, load/total time, prompt/output token counts, processing times, and
tokens/second from the [Ollama API](https://docs.ollama.com/api/generate).
Rates use tokens divided by the corresponding duration. These are measurements
of that request, not model-generated estimates or previous chat statistics.

Generation is capped at **128 output tokens**, so answers may be cut short.
`/perf` alone asks how a computer processes a user request. No further input is
needed while it runs; output appears on completion, with a 120-second response
timeout. Cold starts can take longer. Missing metrics show `N/A`.

**`/read` takes one NVIDIA snapshot when invoked.** It reads utilization, VRAM,
temperature, and power through `nvidia-smi`, then optionally asks the model to
explain them. It captures the local machine before that explanation runs, may
include other workloads, and does not measure the explanation's inference.
There is no background polling. Unavailable sensors show `N/A`; explanations
can be wrong and are labelled separately from measurements.

## Configuration and scripting

Select a model at launch with `--ollama-model MODEL` or `OPENWARP_OLLAMA_MODEL`.
Set the server with `--ollama-url URL` or `OPENWARP_OLLAMA_URL`; this also applies
to `/models` and `/pull`. With a remote server, `/perf` measures remote inference
while `/read` still measures the local GPU. Requests are independent; OpenWarp
does not retain chat history.

```bash
openwarp --read
openwarp --ollama-model qwen2.5:0.5b --perf "Explain how a GPU works."
openwarp --ollama-model qwen2.5:0.5b "Hello"
```

A custom agent can use `--agent-url URL` or `OPENWARP_AGENT_URL`, with optional
`OPENWARP_AGENT_TOKEN` bearer authentication. Its endpoint must accept POST JSON
`{"message":"Hello"}` and return `{"reply":"Hi"}` (30-second timeout).
`/perf` requires Ollama. Packaged installation is available with
`python3 -m pip install -e .` in an environment with pip.

## Troubleshooting

- **Command not found:** check the PATH/launcher setup for `openwarp`; install Ollama for `ollama`.
- **Installer needs zstd:** `sudo apt-get install zstd`, then rerun the installer.
- **Connection refused:** check `/api/tags` using the setup command; start Ollama if needed.
- **Missing model:** `/models` to inspect, `/pull MODEL` to download.
- **GPU read fails:** run `nvidia-smi` in the same Ubuntu/WSL terminal and resolve driver access. AMD/Apple sensor readings are not implemented.
- **Inference stalls:** check the commands below in another terminal. If direct inference also stalls, investigate Ollama before OpenWarp.

```bash
ollama ps
journalctl -u ollama -n 50 --no-pager
ollama run qwen2.5:0.5b "Say hello."
```

Without systemd, inspect the terminal running `ollama serve` instead of the journal.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

GPU/API responses are mocked; these tests do not verify live hardware or Ollama.

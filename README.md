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
GPU detection at install time does not confirm GPU inference; check `ollama ps`
while a model is running.

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
the launcher runs from it. Alternatively, run `python3 -m cli` from the checkout.

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
| `/perf [PROMPT]` | Generate a reply and show its timings; uses a built-in prompt if omitted. |
| `/read` | Capture GPU readings; a connected model also explains them. |
| `/advise` | Check loaded models for partial CPU offload and KV cache fragmentation; suggest fixes. |
| Ordinary text | Chat with the model. |
| `/help` | Show commands. |
| `/exit`, `/quit` | Exit. Ctrl-D or Ctrl-C also exits at the prompt. |

`/gpu` aliases `/read`; `/read explain` explicitly requires a model explanation.

## What gets measured?

**`/perf` runs one request through Ollama** and prints the reply, load/total
time, token counts, durations, and tokens/second as reported by the
[Ollama API](https://docs.ollama.com/api/generate). Output is capped at
**128 tokens**, so answers may be cut short. The request times out after 120
seconds; cold starts can take longer. Missing metrics show `N/A`.

In an interactive session, `/perf` also checks for sustained slowdowns. It
compares each phase's tokens per second with the median of up to 12 earlier
samples, starting after five comparable measurements. Samples are grouped by
Ollama server, model, and token-count ranges. Runs with missing load timings or
load times of one second or more are skipped. A phase gets an alert after three
consecutive comparable samples at least 25% below baseline. Those slow samples
stay out of the baseline. The alert measures throughput; identifying a GPU
bottleneck requires additional profiling.

When a slowdown alert fires, OpenWarp checks Ollama model placement and takes one
post-request NVIDIA GPU snapshot for local servers. For fully GPU-resident
models on local Ollama, it automatically starts a deeper capture and saves it
under `./profiles`. Use `--profile-on-slowdown DIR` to change that location.
Capture requires `nsys` on the Ollama GPU host and `ncu` for kernel counters.
If the Ollama service uses a separate model directory, set
`OLLAMA_MODELS` to the location of the installed model files and ensure your
user can access them.

For GPU-resident models, a fresh loopback Ollama server reruns the triggering
prompt under Nsight Systems and saves the CUDA timeline and kernel summary.
Nsight Compute then captures up to three launches of the kernel with the most
cumulative GPU time and saves memory, compute, occupancy, and scheduler analysis.
These captures include fresh model startup and may contend with the original
server for VRAM. A comparable healthy capture is needed to attribute a
regression to a particular kernel.

**`/read` takes one `nvidia-smi` snapshot** of utilization, VRAM, temperature,
and power, then optionally asks the model to explain it. The snapshot covers the
whole machine, including other workloads, and is taken before the explanation
runs. Unavailable sensors show `N/A`. Explanations can be wrong and are labelled
separately from the readings.

**`/advise` applies fixed rules, not a model.** It reads `/api/ps`, `/api/show`
for each loaded model, and one `nvidia-smi` snapshot (if available). Each rule
that fires prints evidence, a likely cause, fixes, and a source:

- **CPU offload** flags any loaded model with weights on the CPU.
- **KV cache fragmentation** sizes the KV cache Ollama reserved for the context
  window (assuming the default f16 cache type) and compares it with the tokens
  of this session's last `/perf` or chat request. Over 1 GiB reserved and under
  half used suggests PagedAttention (vLLM), which allocates the cache as tokens
  arrive, or a smaller `num_ctx`. Sliding-window and MLA models are skipped.
  `openwarp --advise` has no session request, so this finding is low confidence.

Confidence is high for a measured cause, medium for an inferred one, low for the
fallback. It never changes settings or installs anything. Ollama unloads idle
models after a few minutes, so run it soon after `/perf` or chat.

## Configuration and scripting

Select a model at launch with `--ollama-model MODEL` or `OPENWARP_OLLAMA_MODEL`.
Set the server with `--ollama-url URL` or `OPENWARP_OLLAMA_URL`; this also applies
to `/models` and `/pull`. With a remote server, `/read` still reads the local GPU.
OpenWarp keeps no chat history; each message is sent on its own.

```bash
openwarp --read
openwarp --advise
openwarp --advise --json
openwarp --ollama-model qwen2.5:0.5b --perf "Explain how a GPU works."
openwarp --ollama-model qwen2.5:0.5b "Hello"
```

A custom agent can use `--agent-url URL` or `OPENWARP_AGENT_URL`, with optional
`OPENWARP_AGENT_TOKEN` bearer authentication. Its endpoint must accept POST JSON
`{"message":"Hello"}` and return `{"reply":"Hi"}` (30-second timeout).
`/perf` requires Ollama. To install as a package instead: `python3 -m pip install -e .`

## Troubleshooting

- **Command not found:** check the PATH/launcher setup for `openwarp`; install Ollama for `ollama`.
- **Installer needs zstd:** `sudo apt-get install zstd`, then rerun the installer.
- **Connection refused:** check `/api/tags` using the setup command; start Ollama if needed.
- **Missing model:** `/models` to inspect, `/pull MODEL` to download.
- **GPU read fails:** run `nvidia-smi` in the same Ubuntu/WSL terminal and resolve driver access. AMD/Apple sensor readings are not implemented.
- **Inference stalls:** run these in another terminal. If `ollama run` also stalls, the problem is in Ollama.

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

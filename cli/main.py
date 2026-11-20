"""Standalone OpenWarp command line and interactive shell."""
from __future__ import annotations

import argparse
import os
import sys
import subprocess

from cli import CLI_VERSION
from cli.agent import Agent, AgentError, HttpAgent, OllamaAgent
from cli.gpu import report_gpu
from cli.perf import PerformanceBaseline, report_performance
from cli.advisor import report_advice

HELP = """  OPENWARP · Help
  ────────────────────────────────────────────────────────────
  /read           Show GPU stats; a connected model explains them.
  /perf [PROMPT]  Run your prompt (or a default); show reply + timings.
  /advise         Check loaded models for CPU offload (run /perf or chat first).
  /ollama MODEL   Select an installed model (default: qwen2.5:0.5b).
  /models         List models downloaded in Ollama.
  /pull MODEL     Download a model, then select it for this session.
  Your message    Chat with the model (no GPU stats attached).
  /help           Show commands.
  /exit, /quit    Exit. Ctrl-D / Ctrl-C also exits at the prompt.

  Try: /ollama → /perf   |   Need a model? /pull qwen2.5:0.5b
  /read needs NVIDIA drivers; chat and /perf need Ollama.
"""


def run_ollama(args: list[str], base_url: str) -> bool:
    try:
        return subprocess.run(
            ['ollama', *args], env={**os.environ, 'OLLAMA_HOST': base_url},
            check=False,
        ).returncode == 0
    except OSError as exc:
        raise AgentError('Ollama command unavailable. Install Ollama first.') from exc


def run_shell(agent: Agent | None, ollama_url: str | None = None) -> None:
    base_url = ollama_url or os.environ.get('OPENWARP_OLLAMA_URL', 'http://localhost:11434')
    baseline = PerformanceBaseline()
    print(f"openwarp v{CLI_VERSION} · /help for commands")
    print("/read sends readings to the connected model." if agent is not None
          else "No model selected. /ollama selects Qwen; /models lists installed models.")
    while True:
        try:
            message = input("openwarp > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        command, _, argument = message.partition(' ')
        argument = argument.strip()
        try:
            if message in {"/exit", "/quit"}:
                return
            elif message == "/help":
                print(HELP)
            elif message == "/models":
                if not run_ollama(['list'], base_url):
                    print("Could not list models. Check that Ollama is running.")
            elif command in {"/pull", "/ollama"}:
                model = argument or "qwen2.5:0.5b"
                if command == "/pull" and (model.startswith('-') or any(c.isspace() for c in model)):
                    print("Usage: /pull MODEL")
                    continue
                candidate = OllamaAgent(model, base_url)
                if command == "/pull":
                    print(f"Downloading {model}...", flush=True)
                    if not run_ollama(['pull', model], base_url):
                        print("Download failed. Previous model selection kept.")
                        continue
                agent = candidate
                print(f"Model: {model}. Ready for /perf, /read, or chat.")
            elif command == "/perf":
                if isinstance(agent, OllamaAgent):
                    report_performance(agent, argument or None, baseline)
                else:
                    print("Select an Ollama model first: /ollama MODEL")
            elif message == "/advise":
                report_advice(base_url)
            elif message in {"/read", "/read explain", "/gpu", "/gpu explain"}:
                report_gpu(agent, explain=message.endswith(" explain") or agent is not None)
            elif message.startswith("/"):
                print("Unknown command. Use /help.")
            elif agent is None:
                print("Select a model with /ollama, or download one with /pull MODEL.")
            else:
                print(agent.reply(message))
        except AgentError as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nRequest cancelled.")


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="openwarp", description="A lightweight terminal interface with an optional agent connection.")
    parser.add_argument("--version", action="version", version=f"openwarp {CLI_VERSION}")
    parser.add_argument("--agent-url", default=os.environ.get("OPENWARP_AGENT_URL"),
                        help="HTTP agent endpoint (default: OPENWARP_AGENT_URL)")
    parser.add_argument("--ollama-model", default=os.environ.get("OPENWARP_OLLAMA_MODEL"), help="installed Ollama model for chat and GPU explanations")
    parser.add_argument("--ollama-url", default=os.environ.get("OPENWARP_OLLAMA_URL", "http://localhost:11434"), help="Ollama base URL")
    parser.add_argument("--read", "--gpu", dest="gpu", action="store_true", help="read a GPU snapshot and exit (NVIDIA)")
    parser.add_argument("--perf", action="store_true", help="run a short generation and show Ollama timing metrics")
    parser.add_argument("--advise", action="store_true", help="check loaded Ollama models for CPU offload and exit")
    parser.add_argument("--json", action="store_true", help="with --advise, print the report as JSON")
    parser.add_argument("--explain", action="store_true", help="send the GPU snapshot to the connected model; requires --read")
    parser.add_argument("message", nargs="?", help="send one quoted message and exit")
    args = parser.parse_args(argv)
    if args.advise and (args.gpu or args.perf or args.explain or args.message is not None):
        parser.error("--advise cannot be combined with --read, --perf, --explain, or a message")
    if args.json and not args.advise:
        parser.error("--json requires --advise")
    if args.perf and (args.gpu or args.explain):
        parser.error("--perf cannot be combined with --read or --explain")
    if args.perf and not args.ollama_model:
        parser.error("--perf requires --ollama-model MODEL")
    if args.agent_url and args.ollama_model:
        parser.error("choose either --agent-url or --ollama-model")
    if args.explain and not args.gpu:
        parser.error("--explain requires --read")
    if args.gpu and args.message is not None:
        parser.error("--read cannot be combined with a message")
    try:
        agent = HttpAgent(args.agent_url, os.environ.get("OPENWARP_AGENT_TOKEN")) if args.agent_url else None
        if args.ollama_model:
            agent = OllamaAgent(args.ollama_model, args.ollama_url)
        if args.advise:
            report_advice(args.ollama_url, as_json=args.json)
        elif args.perf:
            report_performance(agent, args.message)
        elif args.gpu:
            report_gpu(agent, args.explain or agent is not None)
        elif args.message is not None:
            if agent is None:
                parser.error("a message requires --ollama-model, --agent-url, or OPENWARP_AGENT_URL")
            print(agent.reply(args.message))
        else:
            run_shell(agent, args.ollama_url)
    except AgentError as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nCancelled.\n")

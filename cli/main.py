"""Standalone OpenWarp command line and interactive shell."""
from __future__ import annotations

import argparse
import os
import sys

from cli import CLI_VERSION
from cli.agent import Agent, AgentError, HttpAgent, OllamaAgent

from cli.gpu import report_gpu

HELP = """  OPENWARP · Help
  ────────────────────────────────────────────────────────────
  /read           Show GPU stats; a connected model explains them.
  /ollama MODEL   Select an installed Ollama model for chat + reads.
  Your message    Chat with the model (no GPU stats attached).
  /help           Show commands.
  /exit, /quit    Exit. Ctrl-D / Ctrl-C also exits at the prompt.

  Try: /ollama MODEL → /read  (find names with ollama list)
  Reads need NVIDIA drivers; model features need Ollama running.
  No model? /read still works. /gpu is an alias for /read.
"""


def run_shell(agent: Agent | None, ollama_url: str | None = None) -> None:
    print(f"openwarp v{CLI_VERSION}")
    print("GPU readings and optional model chat. Type /help for commands.")
    if agent is None:
        print("No agent connected. Use --ollama-model MODEL or --agent-url URL to connect.")
    if agent is not None:
        print("/read sends readings to the connected model.")
    while True:
        try:
            message = input("openwarp > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return
        if message == "/help":
            print(HELP)
            continue
        if message == "/ollama" or message.startswith("/ollama "):
            model = message.partition(" ")[2].strip()
            if not model:
                print("Usage: /ollama MODEL (choose an installed model from ollama list)")
            else:
                try:
                    agent = OllamaAgent(model, ollama_url or os.environ.get("OPENWARP_OLLAMA_URL", "http://localhost:11434"))
                except AgentError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    continue
                print(f"Ollama model: {model}. /read now includes a model explanation.")
            continue
        if message in {"/read", "/read explain", "/gpu", "/gpu explain"}:
            try:
                explain = message.endswith(" explain") or agent is not None
                report_gpu(agent, explain=explain)
            except AgentError as exc:
                print(f"Error: {exc}", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nRequest cancelled.")
            continue
        if message.startswith("/"):
            print("Unknown command. Use /help.")
            continue
        if agent is None:
            print("No agent connected. Use --ollama-model MODEL or --agent-url URL to send messages.")
            continue
        try:
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
    parser.add_argument("--explain", action="store_true", help="send the GPU snapshot to the connected model; requires --read")
    parser.add_argument("message", nargs="?", help="send one quoted message and exit")
    args = parser.parse_args(argv)
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
        if args.gpu:
            explain = args.explain or agent is not None
            report_gpu(agent, explain)
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

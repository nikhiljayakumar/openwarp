"""Standalone OpenWarp command line and interactive shell."""
from __future__ import annotations

import argparse
import os
import sys

from cli import CLI_VERSION
from cli.agent import Agent, AgentError, HttpAgent

HELP = "Type a message to send to the agent. Commands: /help, /exit (or /quit)."


def run_shell(agent: Agent | None) -> None:
    print(f"openwarp v{CLI_VERSION}")
    print(HELP)
    if agent is None:
        print("No agent connected. Set OPENWARP_AGENT_URL or pass --agent-url to connect.")
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
        if message.startswith("/"):
            print("Unknown command. Use /help.")
            continue
        if agent is None:
            print("No agent connected. Use --agent-url URL to send messages.")
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
    parser.add_argument("message", nargs="?", help="send one quoted message and exit")
    args = parser.parse_args(argv)
    try:
        agent = HttpAgent(args.agent_url, os.environ.get("OPENWARP_AGENT_TOKEN")) if args.agent_url else None
        if args.message is not None:
            if agent is None:
                parser.error("a message requires --agent-url or OPENWARP_AGENT_URL")
            print(agent.reply(args.message))
        else:
            run_shell(agent)
    except AgentError as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nCancelled.\n")

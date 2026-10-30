# OpenWarp CLI

A standalone terminal shell with an optional agent API connection. Python 3.11+;
no runtime dependencies.

```sh
python3 -m cli --help
python3 -m cli                 # interactive shell
```

Or install with `python3 -m pip install -e .` and run `openwarp`.
Inside the shell, use `/help`, `/exit`, or `/quit`. Ctrl-D or Ctrl-C exits.

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

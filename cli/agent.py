"""Optional agent connection, independent of the terminal interface.

An endpoint accepts {"message": "..."} and returns {"reply": "..."}.
Provider-specific adapters can implement Agent without changing the CLI.
"""
from __future__ import annotations

import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class Agent(Protocol):
    def reply(self, message: str) -> str: ...


class AgentError(Exception):
    """A user-facing connection or response error."""


class HttpAgent:
    def __init__(self, endpoint: str, token: str | None = None):
        if urlsplit(endpoint).scheme not in {"http", "https"} or not urlsplit(endpoint).netloc:
            raise AgentError("Agent URL must be an absolute HTTP or HTTPS URL.")
        self.endpoint = endpoint
        self.token = token

    def reply(self, message: str) -> str:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.endpoint, data=json.dumps({"message": message}).encode(),
                          headers=headers, method="POST")
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise AgentError(f"Agent returned HTTP {exc.code}.") from exc
        except (URLError, OSError, ValueError) as exc:
            raise AgentError("Could not read a JSON response from the agent.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("reply"), str):
            raise AgentError('Agent response must contain a string field named "reply".')
        return payload["reply"]

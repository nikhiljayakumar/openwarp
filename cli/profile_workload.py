"""Run a single request on an isolated Ollama server under a CUDA profiler."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from cli.agent import AgentError, OllamaAgent
from cli.perf import PERF_OPTIONS


def run(request: dict) -> None:
    host = '127.0.0.1'
    with socket.socket() as listener:
        listener.bind((host, 0))
        port = listener.getsockname()[1]
    address = f'{host}:{port}'
    with subprocess.Popen(['ollama', 'serve'], env={**os.environ, 'OLLAMA_HOST': address}) as server:
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise AgentError('The isolated Ollama server exited during startup.')
                try:
                    with socket.create_connection((host, port), timeout=0.25):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AgentError('The isolated Ollama server did not become ready.')
            agent = OllamaAgent(request['model'], f'http://{address}')
            result = agent.generate(request['prompt'], options=PERF_OPTIONS)
            metrics = {key: value for key, value in result.items()
                       if key.endswith(('_duration', '_count'))}
            print(json.dumps(metrics), flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
    except (AgentError, OSError) as exc:
        sys.exit(str(exc))

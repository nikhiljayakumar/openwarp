import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from urllib.error import URLError

from cli.agent import AgentError, HttpAgent
from cli.main import cli, run_shell


class ShellTests(unittest.TestCase):
    def test_shell_help_empty_input_and_exit(self):
        out = io.StringIO()
        with patch('builtins.input', side_effect=['', '/help', '/exit']), redirect_stdout(out):
            run_shell(None)
        self.assertIn('openwarp v', out.getvalue())
        self.assertIn('No agent connected', out.getvalue())

    def test_shell_sends_message_and_recovers_after_error(self):
        class FakeAgent:
            def reply(self, message):
                if message == 'first':
                    raise AgentError('unavailable')
                return 'received: ' + message
        out, err = io.StringIO(), io.StringIO()
        with patch('builtins.input', side_effect=['first', 'second', '/quit']), redirect_stdout(out), redirect_stderr(err):
            run_shell(FakeAgent())
        self.assertIn('received: second', out.getvalue())
        self.assertIn('unavailable', err.getvalue())

    def test_one_shot_requires_agent(self):
        with patch.dict('os.environ', {}, clear=True), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli(['hello'])
        self.assertEqual(caught.exception.code, 2)

    def test_eof_exits(self):
        with patch('builtins.input', side_effect=EOFError), redirect_stdout(io.StringIO()):
            run_shell(None)


class AgentTests(unittest.TestCase):
    def test_request_contract(self):
        response = io.BytesIO(b'{"reply": "hello"}')
        with patch('cli.agent.urlopen', return_value=response) as send:
            self.assertEqual(HttpAgent('http://localhost/chat', 'secret').reply('hi'), 'hello')
        request = send.call_args.args[0]
        self.assertEqual(json.loads(request.data), {'message': 'hi'})
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret')
        self.assertEqual(request.method, 'POST')
        self.assertEqual(send.call_args.kwargs['timeout'], 30)

    def test_malformed_response(self):
        for data in (b'not json', b'[]', b'{"reply": 4}'):
            with self.subTest(data=data), patch('cli.agent.urlopen', return_value=io.BytesIO(data)):
                with self.assertRaises(AgentError):
                    HttpAgent('http://localhost/chat').reply('hi')

    def test_connection_failure(self):
        with patch('cli.agent.urlopen', side_effect=URLError('offline')):
            with self.assertRaises(AgentError):
                HttpAgent('http://localhost/chat').reply('hi')

    def test_invalid_url(self):
        with self.assertRaises(AgentError):
            HttpAgent('file:///tmp/agent')

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


class ReadTests(unittest.TestCase):
    def test_read_in_shell_without_agent(self):
        with patch('builtins.input', side_effect=['/read', '/exit']), patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(None)
        read.assert_called_once_with(None, explain=False)

    def test_read_flag_without_agent(self):
        with patch.dict('os.environ', {}, clear=True), patch('cli.main.report_gpu') as read:
            cli(['--read'])
        read.assert_called_once_with(None, False)

    def test_read_explain_in_shell(self):
        agent = object()
        with patch('builtins.input', side_effect=['/read explain', '/exit']), patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(agent)
        read.assert_called_once_with(agent, explain=True)

    def test_connected_read_explains(self):
        agent = object()
        with patch('builtins.input', side_effect=['/read', '/exit']), patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(agent)
        self.assertEqual(read.call_args_list[0].kwargs, {'explain': True})

    def test_select_ollama_interactively(self):
        with patch('builtins.input', side_effect=['/ollama local-model', '/read', '/exit']), patch('cli.main.OllamaAgent') as model, patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(None, 'http://localhost:11434')
        model.assert_called_once_with('local-model', 'http://localhost:11434')
        read.assert_called_once_with(model.return_value, explain=True)

    def test_ollama_environment_default(self):
        with patch.dict('os.environ', {'OPENWARP_OLLAMA_MODEL': 'local-model'}, clear=True), patch('cli.main.OllamaAgent') as model, patch('cli.main.report_gpu') as read:
            cli(['--read'])
        read.assert_called_once_with(model.return_value, True)

    def test_deep_command_removed(self):
        with patch('builtins.input', side_effect=['/read deep', '/exit']), patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()) as out:
            run_shell(None)
        read.assert_not_called()
        self.assertIn('Unknown command', out.getvalue())

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from urllib.error import URLError

from cli.agent import AgentError, HttpAgent, OllamaAgent
from cli.main import cli, run_shell


class ShellTests(unittest.TestCase):
    def test_shell_banner_says_no_model_and_skips_empty_input(self):
        out = io.StringIO()
        with patch('builtins.input', side_effect=['', '/help', '/exit']), redirect_stdout(out):
            run_shell(None)
        self.assertIn('openwarp v', out.getvalue())
        self.assertIn('No model selected', out.getvalue())

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

    def test_one_shot_message_without_agent_exits_2(self):
        with patch.dict('os.environ', {}, clear=True), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli(['hello'])
        self.assertEqual(caught.exception.code, 2)

    def test_eof_exits_shell(self):
        with patch('builtins.input', side_effect=EOFError), redirect_stdout(io.StringIO()):
            run_shell(None)


class AgentTests(unittest.TestCase):
    def test_http_agent_posts_json_with_bearer_token(self):
        response = io.BytesIO(b'{"reply": "hello"}')
        with patch('cli.agent.urlopen', return_value=response) as send:
            self.assertEqual(HttpAgent('http://localhost/chat', 'secret').reply('hi'), 'hello')
        request = send.call_args.args[0]
        self.assertEqual(json.loads(request.data), {'message': 'hi'})
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret')
        self.assertEqual(request.method, 'POST')
        self.assertEqual(send.call_args.kwargs['timeout'], 30)

    def test_malformed_reply_raises(self):
        for data in (b'not json', b'[]', b'{"reply": 4}'):
            with self.subTest(data=data), patch('cli.agent.urlopen', return_value=io.BytesIO(data)):
                with self.assertRaises(AgentError):
                    HttpAgent('http://localhost/chat').reply('hi')

    def test_connection_failure_raises(self):
        with patch('cli.agent.urlopen', side_effect=URLError('offline')):
            with self.assertRaises(AgentError):
                HttpAgent('http://localhost/chat').reply('hi')

    def test_non_http_url_rejected(self):
        with self.assertRaises(AgentError):
            HttpAgent('file:///tmp/agent')


class ShellCommandTests(unittest.TestCase):
    def test_read_without_agent_skips_explanation(self):
        with patch('builtins.input', side_effect=['/read', '/exit']), patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(None)
        read.assert_called_once_with(None, explain=False)

    def test_read_flag_without_agent_skips_explanation(self):
        with patch.dict('os.environ', {}, clear=True), patch('cli.main.report_gpu') as read:
            cli(['--read'])
        read.assert_called_once_with(None, False)

    def test_read_explain_without_agent_requests_explanation(self):
        with patch('builtins.input', side_effect=['/read explain', '/exit']), patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(None)
        read.assert_called_once_with(None, explain=True)

    def test_select_ollama_interactively(self):
        with patch('builtins.input', side_effect=['/ollama local-model', '/read', '/exit']), patch('cli.main.OllamaAgent') as model, patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(None, 'http://localhost:11434')
        model.assert_called_once_with('local-model', 'http://localhost:11434')
        read.assert_called_once_with(model.return_value, explain=True)

    def test_ollama_model_env_var_enables_explanation(self):
        with patch.dict('os.environ', {'OPENWARP_OLLAMA_MODEL': 'local-model'}, clear=True), patch('cli.main.OllamaAgent') as model, patch('cli.main.report_gpu') as read:
            cli(['--read'])
        read.assert_called_once_with(model.return_value, True)

    def test_unknown_read_subcommand_is_rejected(self):
        with patch('builtins.input', side_effect=['/read deep', '/exit']), patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()) as out:
            run_shell(None)
        read.assert_not_called()
        self.assertIn('Unknown command', out.getvalue())

    def test_default_model_selected_inside_shell(self):
        with patch('builtins.input', side_effect=['/ollama', '/read', '/exit']), patch('cli.main.OllamaAgent') as model, patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
            run_shell(None, 'http://localhost:11434')
        model.assert_called_once_with('qwen2.5:0.5b', 'http://localhost:11434')
        read.assert_called_once_with(model.return_value, explain=True)

    def test_download_selects_model_only_on_success(self):
        for code in (0, 1):
            with self.subTest(code=code), patch('builtins.input', side_effect=['/pull small-model', '/read', '/exit']), patch('cli.main.OllamaAgent') as model, patch('cli.main.subprocess.run') as run, patch('cli.main.report_gpu') as read, redirect_stdout(io.StringIO()):
                run.return_value.returncode = code
                run_shell(None, 'http://localhost:11434')
                self.assertEqual(run.call_args.args[0], ['ollama', 'pull', 'small-model'])
                read.assert_called_once_with(model.return_value if code == 0 else None, explain=code == 0)

    def test_perf_prompt_inside_shell(self):
        agent = OllamaAgent('small')
        with patch('builtins.input', side_effect=['/perf Explain gravity in one sentence.', '/perf', '/exit']), patch('cli.main.report_performance') as perf, redirect_stdout(io.StringIO()):
            run_shell(agent)
        first, second = perf.call_args_list
        self.assertEqual(first.args[:2], (agent, 'Explain gravity in one sentence.'))
        self.assertEqual(second.args[:2], (agent, None))
        self.assertIs(first.args[2], second.args[2])

    def test_perf_prompt_from_flag(self):
        with patch.dict('os.environ', {}, clear=True), patch('cli.main.OllamaAgent') as model, patch('cli.main.report_performance') as perf:
            cli(['--ollama-model', 'small', '--perf', 'Explain gravity.'])
        perf.assert_called_once_with(model.return_value, 'Explain gravity.')

    def test_missing_ollama_command_keeps_shell_open(self):
        with patch('builtins.input', side_effect=['/models', '/help', '/exit']), patch('cli.main.subprocess.run', side_effect=FileNotFoundError), redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            run_shell(None)
        self.assertIn('Install Ollama', err.getvalue())
        self.assertIn('OPENWARP · Help', out.getvalue())

    def test_cancelled_request_keeps_shell_open(self):
        with patch('builtins.input', side_effect=['/read', '/help', '/exit']), patch('cli.main.report_gpu', side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()) as out:
            run_shell(None)
        self.assertIn('Request cancelled', out.getvalue())
        self.assertIn('OPENWARP · Help', out.getvalue())

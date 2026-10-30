import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch, Mock

from cli.agent import AgentError, OllamaAgent
from cli.gpu import snapshot, report_gpu


class GpuTests(unittest.TestCase):
    def test_multiple_gpus_and_unavailable_reading(self):
        output = '0, NVIDIA RTX, 20, 1000, 8000, 50, 75.5\n1, Other, 0, 0, 4000, 30, [N/A]\n'
        with patch('cli.gpu.subprocess.run', return_value=subprocess.CompletedProcess([], 0, output)) as run:
            data = snapshot()
        self.assertEqual(len(data['gpus']), 2)
        self.assertEqual(data['gpus'][0]['memory_used_mib'], '1000')
        self.assertEqual(data['gpus'][1]['power_w'], '[N/A]')
        self.assertIn('--format=csv,noheader,nounits', run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs['timeout'], 10)

    def test_unavailable_or_bad_output(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 10), subprocess.CalledProcessError(1, 'nvidia-smi')):
            with self.subTest(error=error), patch('cli.gpu.subprocess.run', side_effect=error):
                with self.assertRaises(AgentError):
                    snapshot()
        for output in ('', '0, bad\n'):
            with patch('cli.gpu.subprocess.run', return_value=subprocess.CompletedProcess([], 0, output)):
                with self.assertRaises(AgentError):
                    snapshot()

    def test_explanation_receives_real_snapshot(self):
        agent = Mock()
        agent.reply.return_value = 'Explanation'
        with patch('cli.gpu.snapshot', return_value={'gpus': [{'utilization_percent': '42'}]}), redirect_stdout(io.StringIO()) as out:
            report_gpu(agent, True)
        self.assertIn('42', agent.reply.call_args.args[0])
        self.assertIn('Explanation', out.getvalue())
        self.assertIn('42', out.getvalue())

    def test_explanation_requires_agent(self):
        with patch('cli.gpu.snapshot') as read:
            with self.assertRaises(AgentError):
                report_gpu(None, True)
        read.assert_not_called()

    def test_native_ollama_contract(self):
        with patch('cli.agent.urlopen', return_value=io.BytesIO(b'{"response":"ok"}')) as send:
            self.assertEqual(OllamaAgent('local-model').reply('hello'), 'ok')
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, 'http://localhost:11434/api/generate')
        self.assertEqual(json.loads(request.data), {'model': 'local-model', 'prompt': 'hello', 'stream': False})

    def test_bad_ollama_response(self):
        with patch('cli.agent.urlopen', return_value=io.BytesIO(b'{"error":"missing model"}')):
            with self.assertRaises(AgentError):
                OllamaAgent('local-model').reply('hello')

    def test_pretty_output_with_units_and_unavailable_sensor(self):
        data = {'captured_at': '2026-10-30T12:00:00+00:00', 'source': 'nvidia-smi', 'gpus': [
            dict(index='0', name='NVIDIA RTX', utilization_percent='42', memory_used_mib='1000',
                 memory_total_mib='8000', temperature_c='50', power_w='[N/A]'),
            dict(index='1', name='Second GPU', utilization_percent='0', power_w='75.5'),
        ]}
        with patch('cli.gpu.snapshot', return_value=data), redirect_stdout(io.StringIO()) as out:
            report_gpu()
        text = out.getvalue()
        for expected in ('GPU 0 · NVIDIA RTX', 'GPU 1 · Second GPU', '42%', '1000 MiB / 8000 MiB', '50 °C', '75.5 W', 'Power         N/A'):
            self.assertIn(expected, text)
        self.assertNotIn('[N/A] W', text)
        self.assertNotIn('"gpus"', text)

    def test_ollama_request_contains_visible_gpu_card(self):
        data = {'captured_at': '2026-10-30T20:40:18+00:00', 'source': 'nvidia-smi', 'gpus': [
            dict(index='0', name='NVIDIA GeForce RTX 4070 Laptop GPU', utilization_percent='0',
                 memory_used_mib='605', memory_total_mib='8188', temperature_c='46', power_w='5.01'),
        ]}
        with patch('cli.gpu.snapshot', return_value=data), patch('cli.agent.urlopen', return_value=io.BytesIO(b'{"response":"Low current utilization."}')) as send, redirect_stdout(io.StringIO()) as out:
            report_gpu(OllamaAgent('qwen2.5:0.5b'), True)
        prompt = json.loads(send.call_args.args[0].data)['prompt']
        for value in ('NVIDIA GeForce RTX 4070 Laptop GPU', '0%', '605 MiB / 8188 MiB', '46 °C', '5.01 W'):
            self.assertIn(value, prompt)
            self.assertIn(value, out.getvalue())
        self.assertLess(prompt.index('605 MiB'), prompt.index('Explain these readings'))

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from cli.agent import OllamaAgent, AgentError
from cli.perf import PerformanceBaseline, format_performance, report_performance


class PerformanceTests(unittest.TestCase):
    def test_baseline_flags_sustained_generation_slowdown_once(self):
        baseline = PerformanceBaseline()
        agent = OllamaAgent('small')
        normal = {'load_duration': 10, 'prompt_eval_count': 40, 'prompt_eval_duration': 100_000_000,
                  'eval_count': 40, 'eval_duration': 1_000_000_000}
        slow = {**normal, 'eval_duration': 2_000_000_000}
        for _ in range(5):
            self.assertEqual(baseline.observe(agent, normal), [])
        for _ in range(3):
            self.assertEqual(baseline.observe(agent, {**slow, 'load_duration': 2_000_000_000}), [])
            self.assertEqual(baseline.observe(agent, {**slow, 'prompt_eval_count': 100}), [])
        self.assertEqual(baseline.observe(agent, slow), [])
        self.assertEqual(baseline.observe(agent, normal), [])
        for _ in range(2):
            self.assertEqual(baseline.observe(agent, slow), [])
        alert, = baseline.observe(agent, slow)
        self.assertIn('Generation slowdown: 20.00 tokens/s vs 40.00 baseline', str(alert))
        self.assertEqual(baseline.observe(agent, slow), [])

    def test_rates_are_tokens_per_duration(self):
        text = format_performance({'model': 'small', 'total_duration': 3000000000,
                                  'eval_count': 40, 'eval_duration': 2000000000,
                                  'prompt_eval_count': 10, 'prompt_eval_duration': 500000000})
        self.assertIn('20.00 tokens/s', text)
        self.assertIn('3.000 s', text)
        self.assertIn('Load time       N/A', text)

    def test_missing_or_zero_values_show_na(self):
        text = format_performance({'eval_count': 10, 'eval_duration': 0})
        self.assertIn('Generate rate   N/A', text)
        self.assertIn('Prompt tokens   N/A', text)

    def test_default_run_caps_output_at_128_tokens(self):
        result = {'response': 'hello', 'model': 'small', 'eval_count': 20, 'eval_duration': 1000000000}
        with patch('cli.agent.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as send, redirect_stdout(io.StringIO()) as out:
            report_performance(OllamaAgent('small'))
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body['options']['num_predict'], 128)
        self.assertEqual(body['stream'], False)
        self.assertIn('20.00 tokens/s', out.getvalue())

    def test_timeout_names_120_second_limit(self):
        with patch('cli.agent.urlopen', side_effect=TimeoutError()):
            with self.assertRaisesRegex(AgentError, '120-second'):
                OllamaAgent('small').generate('hello')

    def test_custom_prompt_is_sent_and_reply_printed(self):
        result = {'response': 'A custom answer', 'eval_count': 3, 'eval_duration': 1000000000}
        with patch('cli.agent.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as send, redirect_stdout(io.StringIO()) as out:
            report_performance(OllamaAgent('small'), 'Why is the sky blue?')
        self.assertEqual(json.loads(send.call_args.args[0].data)['prompt'], 'Why is the sky blue?')
        self.assertIn('A custom answer', out.getvalue())
        self.assertIn('3.00 tokens/s', out.getvalue())

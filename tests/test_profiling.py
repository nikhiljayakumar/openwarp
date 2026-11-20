from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from cli.advisor.snapshot import LoadedModel, Snapshot
from cli.agent import OllamaAgent
from cli.perf import PerformanceBaseline, report_performance
from cli.profiling import _capture, investigate_slowdown


class ProfilingTests(unittest.TestCase):
    def test_perf_calls_followup_once_when_both_phases_trigger(self):
        agent = OllamaAgent('small')
        baseline = PerformanceBaseline()
        normal = {'response': 'ok', 'load_duration': 1, 'prompt_eval_count': 40,
                  'eval_count': 40, 'prompt_eval_duration': 100, 'eval_duration': 100}
        slow = {**normal, 'eval_duration': 200, 'prompt_eval_duration': 200}
        followup = Mock()
        with patch.object(agent, 'generate', side_effect=[normal] * 5 + [slow] * 4):
            with redirect_stdout(StringIO()):
                for _ in range(7):
                    report_performance(agent, 'sample', baseline, on_slowdown=followup)
                followup.assert_not_called()
                for _ in range(2):
                    report_performance(agent, 'sample', baseline, on_slowdown=followup)
        followup.assert_called_once_with(agent, 'sample')

    def test_remote_placement_skips_local_gpu_and_capture(self):
        agent = OllamaAgent('small', 'http://example.test:11434')
        with patch('cli.profiling.collect', return_value=Snapshot(())) as collect:
            with patch('cli.profiling._capture') as capture, redirect_stdout(StringIO()):
                investigate_slowdown(agent, 'sample', Path('profiles'))
        collect.assert_called_once_with('http://example.test:11434', include_local_gpu=False)
        capture.assert_not_called()

    def test_offloaded_model_skips_counter_capture(self):
        agent = OllamaAgent('small')
        snapshot = Snapshot((LoadedModel('small:latest', 100, 40),))
        with patch('cli.profiling.collect', return_value=snapshot):
            with patch('cli.profiling._capture') as capture, redirect_stdout(StringIO()):
                investigate_slowdown(agent, 'sample', Path('profiles'))
        capture.assert_not_called()

    def test_gpu_resident_model_captures_with_default_output(self):
        agent = OllamaAgent('small')
        snapshot = Snapshot((LoadedModel('small:latest', 100, 100),))
        with patch('cli.profiling.collect', return_value=snapshot):
            with patch('cli.profiling._capture') as capture, redirect_stdout(StringIO()):
                investigate_slowdown(agent, 'sample')
        capture.assert_called_once_with(agent, 'sample', Path('profiles'))

    def test_trace_selects_kernel_by_aggregate_time(self):
        def run(command, log, timeout=180):
            if command[:2] == ['nsys', 'stats']:
                log.write_text('Progress\nTotal Time (ns),Name\n10,first\n900,selected\n')
            if '--export' in command:
                path = Path(command[command.index('--export') + 1] + '.ncu-rep')
                path.touch()

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            with patch('cli.profiling.shutil.which', side_effect=lambda tool: tool):
                with patch('cli.profiling._run', side_effect=run) as invoke:
                    _capture(OllamaAgent('small'), 'sample', Path(directory))
        trace, summary, counters, analysis = (call.args[0] for call in invoke.call_args_list)
        self.assertIn('cli.profile_workload', trace)
        self.assertIn('cuda_gpu_kern_sum:mangled', summary)
        self.assertIn('regex:^selected$', counters)
        self.assertIn('details', analysis)

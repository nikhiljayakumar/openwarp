import io
import json
import os
import re
import subprocess
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from cli.agent import AgentError
from cli.advisor import report_advice
from cli.advisor.collect import collect, ollama_placement
from cli.advisor.engine import RuleEngine
from cli.advisor.model import Citation, Confidence, Finding, FixOption
from cli.advisor.render import render_json, render_text
from cli.advisor.rule import Rule
from cli.advisor.rules import DEFAULT_RULES
from cli.advisor.rules.cpu_offload import CpuOffloadRule
from cli.advisor.snapshot import GpuReading, LoadedModel, Snapshot
from cli.main import cli, run_shell

GB = 10**9
MIB = 1024 * 1024

# Real measurement: qwen2.5:14b, ollama ps showed 7%/93% CPU/GPU, 32768 context.
QWEN_14B = LoadedModel('qwen2.5:14b', 15_700_000_000, 14_600_000_000, 32768)
FIXTURES = Path(__file__).parent / 'fixtures'


def gpu(used_mib):
    return (GpuReading(0, 'NVIDIA RTX', used_mib, 16376),)


def snap(*models, other_mib=None):
    """Snapshot whose GPU shows the models' size_vram plus other_mib (None: no nvidia-smi)."""
    gpus = () if other_mib is None else gpu(round(sum(m.size_vram for m in models) / MIB) + other_mib)
    return Snapshot(models, gpus)


def offload(*models, other_mib=None):
    return CpuOffloadRule().evaluate(snap(*models, other_mib=other_mib))


def evidence(finding):
    return {e.label: (e.value, e.note) for e in finding.evidence}


def all_commands(finding):
    return [cmd for fix in finding.fixes for cmd in fix.commands]


class CpuOffloadRuleTests(unittest.TestCase):
    def test_model_fully_on_gpu_has_no_finding(self):
        self.assertEqual(offload(LoadedModel('small', 2 * GB, 2 * GB, 4096), other_mib=500), [])

    def test_measured_numbers_appear_once(self):
        [finding] = offload(QWEN_14B, other_mib=2315)
        measured = ('7%', '93%', '1.1 GB', '15.7 GB', '32,768', '32768', '32k', '2,315', '2315')
        texts = [finding.summary, finding.cause, finding.explanation, finding.confidence.basis,
                 *(x for e in finding.evidence for x in (e.value, e.note or '')),
                 *(x for fix in finding.fixes for x in (fix.label, fix.note or '', *fix.commands))]
        for number in measured:
            with self.subTest(number=number):
                self.assertLessEqual(sum(t.count(number) for t in texts), 1)

    def test_wsl_idle_desktop_still_blames_kv_cache(self):
        # Windows holds 2315 MiB at idle; the old order + 1024 MiB threshold blamed other apps.
        [finding] = offload(QWEN_14B, other_mib=2315)
        self.assertEqual((finding.rule_id, finding.subject), ('cpu-offload', 'qwen2.5:14b'))
        self.assertEqual(evidence(finding)['Other VRAM'], ('2,315 MiB', None))
        self.assertEqual(finding.cause, 'KV cache for the long context')
        self.assertEqual(finding.confidence.level, 'high')

    def test_kv_cache_outranks_other_vram_over_threshold(self):
        [finding] = offload(QWEN_14B, other_mib=5000)
        self.assertEqual(finding.cause, 'KV cache for the long context')

    def test_multiple_loaded_models_outrank_kv_cache(self):
        [finding] = offload(QWEN_14B, LoadedModel('qwen2.5:0.5b', GB, GB, 4096), other_mib=5000)
        self.assertEqual(finding.cause, 'Several loaded models competing for VRAM')
        self.assertEqual(finding.confidence.level, 'high')
        self.assertEqual(evidence(finding)['Loaded models'], ('2', None))
        self.assertIn('ollama stop qwen2.5:0.5b', all_commands(finding))
        self.assertNotIn('ollama stop qwen2.5:14b', all_commands(finding))
        self.assertIn('  Environment="OLLAMA_MAX_LOADED_MODELS=1"', all_commands(finding))

    def test_other_vram_over_threshold_blames_other_apps(self):
        [finding] = offload(LoadedModel('big', 10 * GB, 8 * GB, 4096), other_mib=3500)
        self.assertEqual(finding.cause, 'Other GPU apps holding VRAM')
        self.assertEqual(evidence(finding)['Other VRAM'], ('3,500 MiB', 'above the 3,072 MiB threshold'))
        self.assertEqual(evidence(finding)['Context'], ('4,096 tokens', None))
        self.assertEqual(finding.confidence.level, 'medium')

    def test_other_vram_under_threshold_falls_back_low(self):
        [finding] = offload(LoadedModel('big', 10 * GB, 8 * GB, 4096), other_mib=3000)
        self.assertEqual(finding.cause, 'Weights larger than free VRAM')
        self.assertEqual(finding.fixes[0].note, 'e.g. Q4_K_M')
        self.assertEqual(finding.confidence.level, 'low')
        self.assertEqual(evidence(finding)['Placement'], ('20%/80% CPU/GPU', '2.0 GB of 10.0 GB on CPU'))

    def test_unknown_gpu_readings_fall_back_to_weights(self):
        [finding] = offload(LoadedModel('big', 10 * GB, 8 * GB), other_mib=None)
        self.assertEqual(finding.cause, 'Weights larger than free VRAM')
        self.assertEqual(evidence(finding)['Other VRAM'], ('unknown', 'GPU snapshot unavailable'))
        self.assertEqual(evidence(finding)['Context'], ('unknown', None))
        unknown_reading = Snapshot((QWEN_14B,), (GpuReading(0, 'x', None, None),))
        [finding] = CpuOffloadRule().evaluate(unknown_reading)
        self.assertEqual(evidence(finding)['Other VRAM'], ('unknown', 'GPU snapshot unavailable'))


class FakeRule(Rule):
    citation = Citation('a', 't', 'v', 2000)

    def __init__(self, rule_id, applies=True):
        self.id, self._applies = rule_id, applies

    def applies(self, snap):
        return self._applies

    def skip_reason(self, snap):
        return f'{self.id} needs something else'

    def evaluate(self, snap):
        return [Finding(self.id, 'Fake', 'm', 's.', (), 'c', (FixOption('f', ()),), 'e',
                        Confidence('low', 'b'), self.citation)]


class EngineTests(unittest.TestCase):
    def test_runs_rules_in_order(self):
        report = RuleEngine([FakeRule('second'), FakeRule('first')]).run(snap(QWEN_14B))
        self.assertEqual([f.rule_id for f in report.findings], ['second', 'first'])

    def test_json_is_deterministic_and_sorted(self):
        s = snap(QWEN_14B, other_mib=2315)
        first = render_json(RuleEngine(DEFAULT_RULES).run(s))
        self.assertEqual(first, render_json(RuleEngine(DEFAULT_RULES).run(s)))
        data = json.loads(first)
        f = data['findings'][0]
        self.assertEqual(f['confidence']['level'], 'high')
        self.assertEqual(f['evidence'][0], {'label': 'Placement', 'value': '7%/93% CPU/GPU',
                                            'note': '1.1 GB of 15.7 GB on CPU'})
        self.assertEqual(f['fixes'][0]['commands'], ['/set parameter num_ctx 8192'])
        self.assertEqual(f['citation']['arxiv'], '2312.12456')
        for key in ('title', 'subject', 'summary', 'cause', 'explanation'):
            self.assertIn(key, f)
        self.assertEqual(first, json.dumps(data, indent=2, sort_keys=True))

    def test_not_applicable_rule_is_skipped(self):
        report = RuleEngine([FakeRule('off', applies=False), FakeRule('on')]).run(snap(QWEN_14B))
        self.assertEqual(report.skipped, (('off', 'off needs something else'),))
        self.assertEqual([f.rule_id for f in report.findings], ['on'])
        self.assertIn('Skipped off: off needs something else', render_text(report, width=80, color=False))

    def test_cpu_offload_skipped_without_loaded_models(self):
        report = RuleEngine(DEFAULT_RULES).run(snap())
        self.assertEqual(report.findings, ())
        self.assertEqual(report.skipped[0][0], 'cpu-offload')
        self.assertIn('Run /perf or chat first', render_text(report))

    def test_all_models_on_gpu_prints_passed_message(self):
        report = RuleEngine(DEFAULT_RULES).run(snap(LoadedModel('s', GB, GB), other_mib=0))
        self.assertEqual(render_text(report), 'No offload detected: all loaded models are 100% on GPU.')


SCENARIOS = {
    'kv_cache': snap(QWEN_14B, other_mib=2315),
    'multiple_models': snap(QWEN_14B, LoadedModel('qwen2.5:0.5b', GB, GB, 4096), other_mib=5000),
    'other_apps': snap(LoadedModel('big', 10 * GB, 8 * GB, 4096), other_mib=3500),
    'weights': snap(LoadedModel('big', 10 * GB, 8 * GB), other_mib=None),
}


class RenderTextTests(unittest.TestCase):
    def test_text_matches_golden_fixture(self):
        report = RuleEngine(DEFAULT_RULES).run(SCENARIOS['kv_cache'])
        expected = (FIXTURES / 'advise_cpu_offload.txt').read_text(encoding='utf-8')
        self.assertEqual(render_text(report, width=80, color=False) + '\n', expected)

    def test_lines_fit_width_except_unbreakable(self):
        for name, s in SCENARIOS.items():
            report = RuleEngine(DEFAULT_RULES).run(s)
            unbreakable = {str(f.citation) for f in report.findings}
            unbreakable |= {cmd for f in report.findings for cmd in all_commands(f)}
            for width in (40, 60, 80, 100):
                for line in render_text(report, width=width, color=False).splitlines():
                    with self.subTest(scenario=name, width=width, line=line):
                        if len(line) > width:
                            self.assertTrue(any(line.endswith(u) for u in unbreakable))

    def test_commands_and_citation_never_split(self):
        for name, s in SCENARIOS.items():
            report = RuleEngine(DEFAULT_RULES).run(s)
            lines = render_text(report, width=40, color=False).splitlines()
            for f in report.findings:
                for u in (str(f.citation), *all_commands(f)):
                    with self.subTest(scenario=name, text=u):
                        self.assertTrue(any(line.endswith(u) for line in lines))

    def test_default_width_is_terminal_capped_at_100(self):
        report = RuleEngine(DEFAULT_RULES).run(SCENARIOS['kv_cache'])
        with patch('cli.advisor.render.shutil.get_terminal_size', return_value=os.terminal_size((300, 50))):
            wide = render_text(report, color=False)
        self.assertEqual(wide, render_text(report, width=100, color=False))
        with patch('cli.advisor.render.shutil.get_terminal_size', return_value=os.terminal_size((72, 50))):
            self.assertEqual(render_text(report, color=False), render_text(report, width=72, color=False))

    def test_color_only_on_tty_without_no_color(self):
        report = RuleEngine(DEFAULT_RULES).run(SCENARIOS['kv_cache'])
        cases = ((True, {}, True), (True, {'NO_COLOR': '1'}, False), (False, {}, False))
        for tty, env, colored in cases:
            with self.subTest(tty=tty, env=env), patch.dict('os.environ', env, clear=True), \
                 patch('cli.advisor.render.sys.stdout') as stdout:
                stdout.isatty.return_value = tty
                text = render_text(report, width=80)
            self.assertEqual('\033[' in text, colored)
            if colored:
                self.assertIn('\033[31mconfidence: high\033[0m', text)
                self.assertEqual(re.sub(r'\033\[[0-9;]*m', '', text), render_text(report, width=80, color=False))


def ps_response(models):
    return io.BytesIO(json.dumps({'models': models}).encode())


RAW_QWEN = {'name': 'qwen2.5:14b', 'model': 'qwen2.5:14b', 'size': QWEN_14B.size, 'size_vram': QWEN_14B.size_vram,
            'context_length': 32768}


class CollectTests(unittest.TestCase):
    def test_reads_api_ps_and_gpu(self):
        smi = f'0, NVIDIA RTX, 90, {round(QWEN_14B.size_vram / MIB) + 400}, 16376, 60, 120\n'
        with patch('cli.advisor.collect.urlopen', return_value=ps_response([RAW_QWEN, {'name': 'old', 'size': 5, 'size_vram': 5}])) as send, \
             patch('cli.gpu.subprocess.run', return_value=subprocess.CompletedProcess([], 0, smi)):
            s = collect('http://localhost:11434/')
        self.assertEqual(send.call_args.args[0], 'http://localhost:11434/api/ps')
        self.assertEqual(s.models, (QWEN_14B, LoadedModel('old', 5, 5, None)))
        self.assertEqual(s.gpus, (GpuReading(0, 'NVIDIA RTX', round(QWEN_14B.size_vram / MIB) + 400, 16376),))

    def test_missing_nvidia_smi_gives_no_gpus(self):
        with patch('cli.advisor.collect.urlopen', return_value=ps_response([RAW_QWEN])), \
             patch('cli.gpu.subprocess.run', side_effect=FileNotFoundError()):
            self.assertEqual(collect('http://localhost:11434').gpus, ())

    def test_na_gpu_readings_parse_to_none(self):
        smi = '0, NVIDIA RTX, [N/A], [N/A], [N/A], 60, 120\n'
        with patch('cli.advisor.collect.urlopen', return_value=ps_response([])), \
             patch('cli.gpu.subprocess.run', return_value=subprocess.CompletedProcess([], 0, smi)):
            self.assertIsNone(collect('http://localhost:11434').gpus[0].memory_used_mib)

    def test_ollama_failures_raise_agent_error(self):
        for error in (URLError('refused'), HTTPError('u', 500, 'err', {}, io.BytesIO()), TimeoutError(), OSError()):
            with self.subTest(error=error), patch('cli.advisor.collect.urlopen', side_effect=error):
                with self.assertRaises(AgentError):
                    ollama_placement('http://localhost:11434')
        for data in (b'not json', b'[]', b'{"models": [{"name": "x"}]}'):
            with self.subTest(data=data), patch('cli.advisor.collect.urlopen', return_value=io.BytesIO(data)):
                with self.assertRaises(AgentError):
                    ollama_placement('http://localhost:11434')
        with patch('cli.advisor.collect.urlopen', side_effect=URLError('refused')):
            with self.assertRaisesRegex(AgentError, 'Cannot connect to http://localhost:11434/api/ps'):
                ollama_placement('http://localhost:11434')


class AdviseWiringTests(unittest.TestCase):
    def test_report_advice_text_and_json(self):
        for as_json in (False, True):
            with self.subTest(as_json=as_json), \
                 patch('cli.advisor.collect.urlopen', return_value=ps_response([RAW_QWEN])), \
                 patch('cli.gpu.subprocess.run', side_effect=FileNotFoundError()), \
                 redirect_stdout(io.StringIO()) as out:
                report_advice('http://localhost:11434', as_json=as_json)
            if as_json:
                self.assertEqual(json.loads(out.getvalue())['findings'][0]['rule_id'], 'cpu-offload')
            else:
                self.assertIn('KV cache for the long context', out.getvalue())

    def test_shell_advise_uses_shell_ollama_url(self):
        with patch('builtins.input', side_effect=['/advise', '/exit']), patch('cli.main.report_advice') as advise, \
             redirect_stdout(io.StringIO()):
            run_shell(None, 'http://example:11434')
        advise.assert_called_once_with('http://example:11434')

    def test_advise_flags_pass_json_choice(self):
        for argv, as_json in ((['--advise'], False), (['--advise', '--json'], True)):
            with self.subTest(argv=argv), patch.dict('os.environ', {}, clear=True), patch('cli.main.report_advice') as advise:
                cli(argv)
            advise.assert_called_once_with('http://localhost:11434', as_json=as_json)

    def test_advise_flag_conflicts_exit_2(self):
        for argv in (['--advise', '--read'], ['--advise', '--perf', '--ollama-model', 'm'], ['--json']):
            with self.subTest(argv=argv), patch.dict('os.environ', {}, clear=True), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    cli(argv)
                self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

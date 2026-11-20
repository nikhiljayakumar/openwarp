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

from cli.agent import AgentError, HttpAgent, OllamaAgent
from cli.advisor import report_advice
from cli.advisor.collect import collect, ollama_placement
from cli.advisor.engine import RuleEngine
from cli.advisor.model import Citation, Confidence, Finding, FixOption
from cli.advisor.render import render_json, render_text
from cli.advisor.rule import Rule
from cli.advisor.rules import DEFAULT_RULES
from cli.advisor.rules.cpu_offload import CpuOffloadRule
from cli.advisor.rules.kv_fragmentation import KvFragmentationRule
from cli.advisor.snapshot import GpuReading, KvLayout, LoadedModel, MeasuredRequest, Snapshot, parse_kv_layout
from cli.main import cli, last_request, run_shell

GB = 10**9
MIB = 1024 * 1024

# Real measurement: qwen2.5:14b, ollama ps showed 7%/93% CPU/GPU, 32768 context.
QWEN_14B = LoadedModel('qwen2.5:14b', 15_700_000_000, 14_600_000_000, 32768)
# Qwen2.5-14B config: 48 layers, 8 KV heads, 5120 / 40 heads = 128 dims; 192 KiB per token at f16.
QWEN_14B_KV = KvLayout('qwen2.5:14b', 48, 8, 128, 128)
QWEN_SHOW = {'model_info': {'general.architecture': 'qwen2', 'qwen2.block_count': 48,
                            'qwen2.attention.head_count': 40, 'qwen2.attention.head_count_kv': 8,
                            'qwen2.embedding_length': 5120, 'qwen2.context_length': 32768}}
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


def kv(model, layout=QWEN_14B_KV, tokens=None):
    requests = () if tokens is None else (MeasuredRequest(model.name, tokens),)
    return KvFragmentationRule().evaluate(Snapshot((model,), (), (layout,), requests))


class KvFragmentationRuleTests(unittest.TestCase):
    def test_short_measured_request_suggests_paged_attention(self):
        [finding] = kv(QWEN_14B, tokens=158)
        self.assertEqual((finding.rule_id, finding.subject), ('kv-fragmentation', 'qwen2.5:14b'))
        self.assertEqual(evidence(finding)['Per token'], ('192 KiB', 'f16 K and V, 48 layers × 8 KV heads'))
        self.assertEqual(evidence(finding)['KV reserved'], ('6,144 MiB', 'above the 1,024 MiB threshold'))
        self.assertEqual(evidence(finding)['Last request'], ('158 tokens', '0.5% of the context'))
        self.assertEqual(finding.cause, 'Context window far larger than the last request')
        self.assertEqual(finding.confidence.level, 'high')
        self.assertEqual(finding.fixes[0].label, 'PagedAttention (vLLM)')
        self.assertIn('vllm serve HF_MODEL', all_commands(finding))
        self.assertIn('/set parameter num_ctx 2048', all_commands(finding))

    def test_suggested_context_is_next_power_of_two(self):
        [finding] = kv(QWEN_14B, tokens=5000)
        self.assertIn('/set parameter num_ctx 8192', all_commands(finding))

    def test_measured_numbers_appear_once(self):
        [finding] = kv(QWEN_14B, tokens=158)
        measured = ('32,768', '32768', '192', '6,144', '6144', '158', '0.5%')
        texts = [finding.summary, finding.cause, finding.explanation, finding.confidence.basis,
                 *(x for e in finding.evidence for x in (e.value, e.note or '')),
                 *(x for fix in finding.fixes for x in (fix.label, fix.note or '', *fix.commands))]
        for number in measured:
            with self.subTest(number=number):
                self.assertLessEqual(sum(t.count(number) for t in texts), 1)

    def test_unmeasured_request_falls_back_low(self):
        [finding] = kv(QWEN_14B)
        self.assertEqual(evidence(finding)['Last request'], ('not measured', None))
        self.assertEqual(finding.cause, 'Context window likely larger than the requests')
        self.assertEqual(finding.confidence.level, 'low')
        self.assertEqual(all_commands(finding)[-1], '/perf')

    def test_half_used_window_has_no_finding(self):
        self.assertEqual(kv(QWEN_14B, tokens=16385), [])
        self.assertEqual(len(kv(QWEN_14B, tokens=16384)), 1)

    def test_reservation_under_threshold_has_no_finding(self):
        # qwen2.5:0.5b: 24 layers x 2 KV heads x 64 dims, so a 32k window reserves only 384 MiB.
        small = LoadedModel('qwen2.5:0.5b', GB, GB, 32768)
        self.assertEqual(kv(small, KvLayout('qwen2.5:0.5b', 24, 2, 64, 64)), [])

    def test_request_without_tag_matches_latest(self):
        model = LoadedModel('qwen2.5:latest', 15 * GB, 15 * GB, 32768)
        s = Snapshot((model,), (), (KvLayout('qwen2.5:latest', 48, 8, 128, 128),), (MeasuredRequest('qwen2.5', 158),))
        [finding] = KvFragmentationRule().evaluate(s)
        self.assertEqual(finding.confidence.level, 'high')

    def test_no_smaller_context_fix_at_minimum_window(self):
        # llama2:13b keeps full multi-head attention (40 KV heads): 2,048 tokens already reserve 1,600 MiB.
        model = LoadedModel('llama2:13b', 9 * GB, 9 * GB, 2048)
        [finding] = kv(model, KvLayout('llama2:13b', 40, 40, 128, 128), tokens=158)
        self.assertEqual([fix.label for fix in finding.fixes], ['PagedAttention (vLLM)'])

    def test_skipped_without_layout_context_or_models(self):
        rule = KvFragmentationRule()
        self.assertFalse(rule.applies(snap(QWEN_14B)))
        self.assertIn('supported attention layout', rule.skip_reason(snap(QWEN_14B)))
        self.assertFalse(rule.applies(Snapshot((LoadedModel('qwen2.5:14b', GB, GB),), (), (QWEN_14B_KV,))))
        self.assertIn('No models loaded', rule.skip_reason(snap()))


class ParseKvLayoutTests(unittest.TestCase):
    def test_head_dim_defaults_to_width_over_heads(self):
        self.assertEqual(parse_kv_layout('qwen2.5:14b', QWEN_SHOW), QWEN_14B_KV)

    def test_explicit_lengths_and_missing_kv_heads(self):
        info = {**QWEN_SHOW['model_info'], 'qwen2.attention.key_length': 192, 'qwen2.attention.value_length': 128}
        self.assertEqual(parse_kv_layout('m', {'model_info': info}), KvLayout('m', 48, 8, 192, 128))
        del info['qwen2.attention.head_count_kv']  # absent: every head has its own K and V
        self.assertEqual(parse_kv_layout('m', {'model_info': info}), KvLayout('m', 48, 40, 192, 128))

    def test_unsizable_layouts_are_none(self):
        info = QWEN_SHOW['model_info']
        cases = {'not a dict': [], 'no model_info': {}, 'no architecture': {'model_info': {'qwen2.block_count': 48}},
                 'sliding window': {'model_info': {**info, 'qwen2.attention.sliding_window': 4096}},
                 'MLA': {'model_info': {**info, 'qwen2.attention.kv_lora_rank': 512}},
                 'per-layer KV heads': {'model_info': {**info, 'qwen2.attention.head_count_kv': [8] * 48}},
                 'missing layers': {'model_info': {**info, 'qwen2.block_count': None}}}
        for name, payload in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(parse_kv_layout('m', payload))


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

    def test_shared_skip_reason_printed_once(self):
        report = RuleEngine(DEFAULT_RULES).run(snap())
        self.assertEqual([r for r, _ in report.skipped], ['cpu-offload', 'kv-fragmentation'])
        self.assertEqual(render_text(report).count('Run /perf or chat first'), 1)

    def test_all_models_on_gpu_prints_passed_messages(self):
        small = LoadedModel('s', GB, GB, 4096)
        report = RuleEngine(DEFAULT_RULES).run(Snapshot((small,), gpu(1000), (KvLayout('s', 24, 2, 64, 64),)))
        self.assertEqual(render_text(report), 'No offload detected: all loaded models are 100% on GPU.\n'
                         'No KV cache fragmentation: each KV cache is under 1,024 MiB or at least half in use.')

    def test_json_lists_findings_in_rule_order(self):
        data = json.loads(render_json(RuleEngine(DEFAULT_RULES).run(SCENARIOS['kv_measured'])))
        self.assertEqual([f['rule_id'] for f in data['findings']], ['cpu-offload', 'kv-fragmentation'])
        self.assertEqual(data['findings'][1]['citation']['arxiv'], '2309.06180')


SCENARIOS = {
    'kv_cache': snap(QWEN_14B, other_mib=2315),
    'multiple_models': snap(QWEN_14B, LoadedModel('qwen2.5:0.5b', GB, GB, 4096), other_mib=5000),
    'other_apps': snap(LoadedModel('big', 10 * GB, 8 * GB, 4096), other_mib=3500),
    'weights': snap(LoadedModel('big', 10 * GB, 8 * GB), other_mib=None),
    'kv_measured': Snapshot((QWEN_14B,), (), (QWEN_14B_KV,), (MeasuredRequest('qwen2.5:14b', 158),)),
    'kv_unmeasured': Snapshot((QWEN_14B,), (), (QWEN_14B_KV,)),
}


class RenderTextTests(unittest.TestCase):
    def test_text_matches_golden_fixture(self):
        for rule, scenario, fixture in ((CpuOffloadRule(), 'kv_cache', 'advise_cpu_offload.txt'),
                                        (KvFragmentationRule(), 'kv_measured', 'advise_kv_fragmentation.txt')):
            with self.subTest(fixture=fixture):
                report = RuleEngine([rule]).run(SCENARIOS[scenario])
                expected = (FIXTURES / fixture).read_text(encoding='utf-8')
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


def show_response(payload=QWEN_SHOW):
    return io.BytesIO(json.dumps(payload).encode())


RAW_QWEN = {'name': 'qwen2.5:14b', 'model': 'qwen2.5:14b', 'size': QWEN_14B.size, 'size_vram': QWEN_14B.size_vram,
            'context_length': 32768}


class CollectTests(unittest.TestCase):
    def test_reads_api_ps_gpu_and_layouts(self):
        smi = f'0, NVIDIA RTX, 90, {round(QWEN_14B.size_vram / MIB) + 400}, 16376, 60, 120\n'
        responses = [ps_response([RAW_QWEN, {'name': 'old', 'size': 5, 'size_vram': 5}]), show_response(), URLError('gone')]
        with patch('cli.advisor.collect.urlopen', side_effect=responses) as send, \
             patch('cli.gpu.subprocess.run', return_value=subprocess.CompletedProcess([], 0, smi)):
            s = collect('http://localhost:11434/', (MeasuredRequest('qwen2.5:14b', 158),))
        self.assertEqual(send.call_args_list[0].args[0], 'http://localhost:11434/api/ps')
        show = send.call_args_list[1].args[0]
        self.assertEqual((show.full_url, show.method), ('http://localhost:11434/api/show', 'POST'))
        self.assertEqual(json.loads(show.data), {'model': 'qwen2.5:14b'})
        self.assertEqual(s.models, (QWEN_14B, LoadedModel('old', 5, 5, None)))
        self.assertEqual(s.gpus, (GpuReading(0, 'NVIDIA RTX', round(QWEN_14B.size_vram / MIB) + 400, 16376, 90),))
        self.assertEqual(s.kv_layouts, (QWEN_14B_KV,))  # 'old' got no /api/show answer and is left out
        self.assertEqual(s.requests, (MeasuredRequest('qwen2.5:14b', 158),))

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
        advise.assert_called_once_with('http://example:11434', requests=())

    def test_shell_advise_passes_last_measured_request(self):
        generated = {'response': 'hi', 'done': True, 'prompt_eval_count': 30, 'eval_count': 128}
        with patch('builtins.input', side_effect=['hello', '/advise', '/exit']), \
             patch('cli.agent.urlopen', return_value=io.BytesIO(json.dumps(generated).encode())), \
             patch('cli.main.report_advice') as advise, redirect_stdout(io.StringIO()):
            run_shell(OllamaAgent('qwen2.5:14b'), 'http://example:11434')
        advise.assert_called_once_with('http://example:11434', requests=(MeasuredRequest('qwen2.5:14b', 158),))

    def test_no_measured_request_without_ollama_counts(self):
        self.assertEqual(last_request(OllamaAgent('m')), ())
        self.assertEqual(last_request(HttpAgent('http://localhost/chat')), ())
        self.assertEqual(last_request(None), ())

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

from __future__ import annotations

from dataclasses import asdict
import json
import os
import shutil
import sys
import textwrap

from cli.advisor.model import Finding, Report

MAX_WIDTH = 100
MIN_WIDTH = 40
MIN_TEXT = 20  # narrower than this after a hanging indent: wrap below the label instead
CAUSE_LABEL = 'Likely cause'
FOOTER = 'Advice only. OpenWarp did not change any settings.'
BOLD, RED, YELLOW, RESET = '\033[1m', '\033[31m', '\033[33m', '\033[0m'
CONFIDENCE_COLOR = {'high': RED, 'medium': YELLOW, 'low': YELLOW}


def _paint(text: str, code: str, color: bool) -> str:
    return f'{code}{text}{RESET}' if color else text


def _wrap(text: str, width: int, prefix: str, fallback: int = 6) -> list[str]:
    """Wrap text after prefix with a hanging indent at the prefix column.

    If that column leaves under MIN_TEXT characters, the prefix gets its own
    line and the text wraps below it at the fallback indent.
    """
    if prefix.strip() and width - len(prefix) < MIN_TEXT:
        return [prefix.rstrip(), *_wrap(text, width, ' ' * fallback)]
    return textwrap.wrap(text, width, initial_indent=prefix, subsequent_indent=' ' * len(prefix),
                         break_long_words=False, break_on_hyphens=False) or [prefix.rstrip()]


def _finding(f: Finding, width: int, color: bool) -> list[str]:
    head = f'{f.title} · {f.subject}'
    conf = f'confidence: {f.confidence.level}'
    painted = _paint(conf, CONFIDENCE_COLOR[f.confidence.level], color)
    if len(head) + 2 + len(conf) <= width:
        lines = [_paint(head, BOLD, color) + ' ' * (width - len(head) - len(conf)) + painted]
    else:
        lines = [*(_paint(line, BOLD, color) for line in _wrap(head, width, '')), '  ' + painted]
    lines += _wrap(f.summary, width, '  ')

    # Evidence rows and the cause share one value column.
    col = max(4 + max((len(e.label) for e in f.evidence), default=0), 2 + len(CAUSE_LABEL)) + 2
    value_width = max((len(e.value) for e in f.evidence), default=0) + 3
    lines += ['', '  Evidence']
    for e in f.evidence:
        prefix = '    ' + e.label.ljust(col - 4)
        if e.note:
            lines += _wrap(e.note, width, prefix + e.value.ljust(value_width))
        else:
            lines.append((prefix + e.value).rstrip())
    lines += _wrap(f.cause, width, '  ' + CAUSE_LABEL.ljust(col - 2), 4)

    # Commands are never wrapped; they continue at the command column.
    label_width = max((len(x.label) for x in f.fixes), default=0) + 3
    lines += ['', '  Fix (pick one)' if len(f.fixes) > 1 else '  Fix']
    for i, fix in enumerate(f.fixes, 1):
        prefix = f'    {i}. ' + fix.label.ljust(label_width)
        pad = ' ' * len(prefix)
        rows = [(pad if n else prefix) + cmd for n, cmd in enumerate(fix.commands)]
        note = f'({fix.note})' if fix.note else None
        if not rows:
            rows = _wrap(note, width, prefix) if note else [prefix.rstrip()]
        elif note and len(rows) == 1 and len(rows[0]) + 3 + len(note) <= width:
            rows[0] += '   ' + note
        elif note:  # below the commands; under the label if the command column is too narrow
            rows += _wrap(note, width, pad if width - len(pad) >= MIN_TEXT else ' ' * 7)
        lines += rows

    lines += ['', '  Why', *_wrap(f.explanation, width, '    ')]
    lines += ['', f'  Source  {f.citation}']  # never split, even past width
    return lines


def render_text(report: Report, width: int | None = None, color: bool | None = None) -> str:
    if not report.findings:
        # Rules often skip for the same reason (no models loaded): print each message once.
        return '\n'.join(dict.fromkeys(message for _, message in report.skipped + report.passed))
    if width is None:
        width = max(MIN_WIDTH, min(shutil.get_terminal_size().columns, MAX_WIDTH))
    if color is None:
        color = sys.stdout.isatty() and 'NO_COLOR' not in os.environ
    n = len(report.findings)
    lines = [_paint(f'OpenWarp advice · {n} finding{"s" * (n != 1)}', BOLD, color)]
    for f in report.findings:
        lines += ['', *_finding(f, width, color)]
    if report.skipped:
        lines.append('')
        for rule_id, why in report.skipped:
            lines += textwrap.wrap(f'Skipped {rule_id}: {why}', width, subsequent_indent='  ')
    lines += ['', *textwrap.wrap(FOOTER, width)]
    return '\n'.join(lines)


def render_json(report: Report) -> str:
    data = {
        'findings': [asdict(f) for f in report.findings],
        'skipped': [{'rule_id': r, 'reason': why} for r, why in report.skipped],
        'passed': [{'rule_id': r, 'message': msg} for r, msg in report.passed],
    }
    return json.dumps(data, indent=2, sort_keys=True)

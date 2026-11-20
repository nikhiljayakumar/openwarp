from __future__ import annotations

from collections.abc import Iterable

from cli.advisor.model import Report
from cli.advisor.rule import Rule
from cli.advisor.snapshot import Snapshot


class RuleEngine:
    def __init__(self, rules: Iterable[Rule]):
        self.rules = tuple(rules)

    def run(self, snap: Snapshot) -> Report:
        findings, skipped, passed = [], [], []
        for rule in self.rules:  # fixed order
            if not rule.applies(snap):
                skipped.append((rule.id, rule.skip_reason(snap)))
                continue
            found = rule.evaluate(snap)
            findings.extend(found)
            if not found:
                passed.append((rule.id, rule.passed_message))
        return Report(tuple(findings), tuple(skipped), tuple(passed))

"""Deterministic advice: collect a Snapshot, run rules over it, render a Report."""
from cli.advisor.collect import collect
from cli.advisor.engine import RuleEngine
from cli.advisor.render import render_json, render_text
from cli.advisor.rules import DEFAULT_RULES


def report_advice(base_url: str, as_json: bool = False) -> None:
    report = RuleEngine(DEFAULT_RULES).run(collect(base_url))
    print(render_json(report) if as_json else render_text(report), flush=True)

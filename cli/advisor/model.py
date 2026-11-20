from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Citation:
    authors: str
    title: str
    venue: str
    year: int
    arxiv: str | None = None

    def __str__(self) -> str:
        text = f'{self.authors}, "{self.title}," {self.venue} {self.year}'
        return f'{text}, arXiv:{self.arxiv}' if self.arxiv else text


@dataclass(frozen=True)
class Confidence:
    level: Literal['high', 'medium', 'low']
    basis: str


@dataclass(frozen=True)
class Evidence:
    label: str
    value: str
    note: str | None = None


@dataclass(frozen=True)
class FixOption:
    label: str
    commands: tuple[str, ...]  # one line each; rendered verbatim, never wrapped
    note: str | None = None


@dataclass(frozen=True)
class Finding:
    rule_id: str
    title: str
    subject: str
    summary: str  # one short sentence
    evidence: tuple[Evidence, ...]
    cause: str
    fixes: tuple[FixOption, ...]
    explanation: str
    confidence: Confidence
    citation: Citation


@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]
    skipped: tuple[tuple[str, str], ...]  # (rule_id, why it did not apply)
    passed: tuple[tuple[str, str], ...]  # (rule_id, all-clear message)

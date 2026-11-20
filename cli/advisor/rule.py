from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from cli.advisor.model import Citation, Finding
from cli.advisor.snapshot import Snapshot


class Rule(ABC):
    id: ClassVar[str]
    citation: ClassVar[Citation]
    passed_message: ClassVar[str] = 'No issues found.'

    def applies(self, snap: Snapshot) -> bool:
        return bool(snap.models)

    def skip_reason(self, snap: Snapshot) -> str:
        return 'No models loaded. Run /perf or chat first, then /advise.'

    @abstractmethod
    def evaluate(self, snap: Snapshot) -> list[Finding]:
        """Pure: no I/O, same snapshot -> same findings."""

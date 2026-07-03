from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    evidence_ids: tuple[str, ...] = ()


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

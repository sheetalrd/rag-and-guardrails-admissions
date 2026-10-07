from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    guard: str
    allowed: bool
    reason: str

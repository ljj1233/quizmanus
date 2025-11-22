from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Command:
    goto: str | None = None
    update: Dict[str, Any] = field(default_factory=dict)

import json
import sys
import types
from pathlib import Path


def _ensure_stubbed_module(name: str, attrs: dict) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.SimpleNamespace()
        sys.modules[name] = module
    for attr, value in attrs.items():
        if not hasattr(module, attr):
            setattr(module, attr, value)


_ensure_stubbed_module(
    "numpy",
    {
        "seed": lambda *_args, **_kwargs: None,
        "random": types.SimpleNamespace(seed=lambda *_args, **_kwargs: None),
    },
)
_ensure_stubbed_module(
    "torch",
    {
        "manual_seed": lambda *_args, **_kwargs: None,
        "cuda": types.SimpleNamespace(manual_seed_all=lambda *_args, **_kwargs: None),
    },
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.eos_token = ""
        self.pad_token = None

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


_ensure_stubbed_module("transformers", {"AutoTokenizer": _DummyTokenizer})

from main import prepare_quiz_dataset


def test_prepare_quiz_dataset_creates_annotated_copy(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    original = [
        {"id": "1", "query": "What is photosynthesis?"},
        {"id": "2", "query": "Explain cellular respiration."},
    ]
    source.write_text(json.dumps(original), encoding="utf-8")

    save_dir = tmp_path / "quiz_results"
    annotated_data, annotated_path = prepare_quiz_dataset(source, save_dir)

    expected = [
        {"id": "1", "query": "What is photosynthesis?", "quiz_url": str(save_dir / "1.md")},
        {"id": "2", "query": "Explain cellular respiration.", "quiz_url": str(save_dir / "2.md")},
    ]

    assert annotated_data == expected
    assert annotated_path.exists()
    saved = json.loads(annotated_path.read_text(encoding="utf-8"))
    assert saved == expected

    # Ensure the source dataset remains unchanged
    assert json.loads(source.read_text(encoding="utf-8")) == original

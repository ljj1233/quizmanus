import json
from typing import Any, Callable


class ValidationError(ValueError):
    pass


def Field(default: Any = None, *, default_factory: Callable[[], Any] | None = None, alias: str | None = None, **_: Any):
    if default_factory is not None:
        return default_factory()
    return default


class BaseModel:
    class Config:
        populate_by_name = False
        extra = "ignore"

    def __init__(self, **data: Any):
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls, data: dict):
        try:
            return cls(**data)
        except Exception as exc:  # pragma: no cover - compatibility shim
            raise ValidationError(str(exc))

    @classmethod
    def model_validate_json(cls, data: str):
        try:
            parsed = json.loads(data)
        except Exception as exc:  # pragma: no cover - compatibility shim
            raise ValidationError(str(exc))
        return cls.model_validate(parsed)

    def model_dump(self, **_: Any):  # pragma: no cover - shim for compatibility
        return self.__dict__

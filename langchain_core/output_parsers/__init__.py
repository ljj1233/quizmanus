import json
from pydantic import ValidationError


class JsonOutputParser:
    def parse(self, text: str):
        return json.loads(text)


class PydanticOutputParser:
    def __init__(self, pydantic_object):
        self.pydantic_object = pydantic_object

    def parse(self, text: str):
        try:
            return self.pydantic_object.model_validate_json(text)
        except ValidationError:
            raise
        except Exception as exc:
            try:
                data = json.loads(text)
            except Exception:
                raise
            if hasattr(self.pydantic_object, "model_validate"):
                return self.pydantic_object.model_validate(data)
            return self.pydantic_object(**data)

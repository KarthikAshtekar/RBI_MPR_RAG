from __future__ import annotations

import json
import os
from typing import Any


SECRET_ENV_NAMES = ("GROQ_API_KEY", "COHERE_API_KEY", "UNSTRUCTURED_API_KEY")


def contains_key_material(value: Any) -> bool:
    text = json.dumps(value, default=str)
    for name in SECRET_ENV_NAMES:
        secret = os.getenv(name)
        if secret and secret in text:
            return True
    return False

"""JSON Schemas for the phone link (docs/phone-protocol.md)."""

import json
from functools import cache
from importlib import resources


@cache
def load(name: str) -> dict:
    """load("phone_message") or load("laptop_message")."""
    return json.loads(resources.files(__package__).joinpath(f"{name}.schema.json").read_text())

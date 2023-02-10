import json
from pathlib import Path

from virtual_rainforest.core.config import register_schema


@register_schema("plants")
def schema() -> dict:
    """Defines the schema that the plant module configuration should conform to."""

    schema_file = Path(__file__).parent.resolve() / "plants_schema.json"

    with schema_file.open() as f:
        config_schema = json.load(f)

    return config_schema
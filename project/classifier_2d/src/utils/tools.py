import json


def format_tune(section):
    """Parse the [tune] config section into typed Python values using JSON parsing."""
    return {key: json.loads(val) for key, val in section.items()}

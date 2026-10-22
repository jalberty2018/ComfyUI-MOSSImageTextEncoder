import math
import numbers
import re
from decimal import Decimal
from pathlib import Path



TOKEN_KEY = "qwen3vl_4b"
ZERO_WEIGHT_BIAS = -80.0
WEIGHT_PATTERN = re.compile(
    r"\((?P<phrase>[^()]+):(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)


def parse_phrases(text):
    parts, spans = [], []
    cursor = length = 0
    for match in WEIGHT_PATTERN.finditer(text):
        raw_phrase = match.group("phrase")
        phrase = raw_phrase.strip()
        if not phrase:
            raise ValueError("A weighted phrase cannot be empty.")
        decimal_weight = Decimal(match.group("weight"))
        weight = float(decimal_weight)
        if decimal_weight < 0 or not math.isfinite(weight) or (weight == 0 and decimal_weight != 0):
            raise ValueError(f"Weight for {phrase!r} must be a finite, representable number of zero or greater.")
        preceding = text[cursor:match.start()]
        parts.extend((preceding, raw_phrase))
        length += len(preceding)
        spans.append({
            "phrase": phrase,
            "weight": weight,
            "start": length + len(raw_phrase) - len(raw_phrase.lstrip()),
            "end": length + len(raw_phrase.rstrip()),
            "source": match.group(0),
        })
        length += len(raw_phrase)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts), spans





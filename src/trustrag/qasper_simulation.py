"""Shared packet parsing and agreement helpers for the human analysis."""

import json
import re
from collections import Counter

FIELDS = ("relevant", "factual", "complete", "citations_supported", "unsupported_additions")

def read_packet(path):
    match = re.search(
        r'<script id="packet" type="application/json">(.*?)</script>',
        path.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    if not match:
        raise ValueError("The existing blinded packet is missing")
    return json.loads(match.group(1))


def joint_success(rating, outcome):
    return (
        outcome == "answered"
        and all(rating[f] == "yes" for f in FIELDS[:4])
        and rating["unsupported_additions"] == "no"
    )


def fleiss(values):
    if not values:
        return {"observed_pairwise_agreement": None, "fleiss_kappa": None}
    categories = ("yes", "no", "unsure")
    n = len(values[0])
    marginals = Counter(v for row in values for v in row)
    observed = sum(
        sum(c * (c - 1) for c in Counter(row).values()) / (n * (n - 1)) for row in values
    ) / len(values)
    expected = sum((marginals[c] / (len(values) * n)) ** 2 for c in categories)
    return {
        "observed_pairwise_agreement": observed,
        "fleiss_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
    }

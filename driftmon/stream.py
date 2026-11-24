"""Replay of a historical dataset as a simulated production stream.

The stream is a replay: each period's rows become available in chronological
order, and ground-truth outcomes mature only after a fixed delay. Nothing
about this is a live system; the simulation exists to make monitoring and
retraining behavior measurable and reproducible.
"""
from __future__ import annotations


def matured_index(current_index: int, label_delay: int) -> int:
    """Last period index whose outcomes have arrived by current_index.

    A period's outcomes become visible `label_delay` periods later, so at
    period t the newest usable outcomes belong to period t - delay.
    Returns -1 when nothing has matured yet.
    """
    return current_index - label_delay

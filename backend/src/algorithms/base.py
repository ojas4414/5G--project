from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AlgorithmOutput:
    actions: Dict[str, np.ndarray]
    aux: Dict[str, float]


class BaseAllocator:
    def __init__(self, name: str):
        self.name = name

    def reset(self) -> None:
        return None

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        raise NotImplementedError

    def observe(self, state: Dict[str, np.ndarray], metrics: Dict[str, np.ndarray]) -> None:
        return None

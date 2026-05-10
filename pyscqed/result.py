"""Lowest level result data structures"""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvaluationResult:
    data: dict[str, dict[str, np.ndarray]]


@dataclass
class NumericalResult:
    data: np.ndarray
    source: str | None = None


class EigenvalueResult(NumericalResult):
    pass


class EigenvectorResult(NumericalResult):
    pass


class FunctionResult(NumericalResult):
    pass

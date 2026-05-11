"""Lowest level result data structures"""
from dataclasses import dataclass

import numpy as np


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


@dataclass
class EvaluationResult:
    data: dict[str, dict[str, NumericalResult]]
    source: str

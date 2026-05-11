"""Lowest level result data structures"""
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


EvalKeys: TypeAlias = list[tuple[str, str]]


@dataclass
class NumericalResult:
    data: np.ndarray
    axis_keys: EvalKeys | None = None
    source: str | None = None

    @classmethod
    def init_from_keys(cls, axis_keys: EvalKeys, shape: tuple[int], source: str | None = None) -> "NumericalResult":
        eval_dims = len(axis_keys)
        if eval_dims == 1:
            data = np.empty(shape, dtype=object)
        else:
            data = np.empty(eval_dims + shape, dtype=object)
        return cls(data=data, axis_keys=axis_keys, source=source)


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

    def get_keys(self) -> EvalKeys:
        keys = []
        for node in self.data:
            for output in self.data[node]:
                keys.append((node, output))
        return keys


@dataclass
class SweepNumericalResult:
    data: dict[EvalKeys, np.ndarray]
    source: str

    def __getitem__(self, subscript):
        if subscript is None:
            return self.data[("Spectrum", "E")]
        return self.data[subscript]

""""""
import numpy as np

from typing import TypeAlias

from .parameters import ParamCollection


SweepVector: TypeAlias = list[float] | np.ndarray[np.float64]


class SweepSpec:
    def __init__(self, collection: ParamCollection):
        self._collection = collection
        self._sweep_data: dict[str, SweepVector] = {}

    def add(self, name: str, values: SweepVector):
        """Adds a sweep dimension with the specified values."""
        if name in self._sweep_data:
            raise ValueError(f"Parameter \"{name}\" is already in the sweep.")
        self._sweep_data[name] = values

    def get_total_count(self) -> int:
        """Gets the total number of sweep points."""
        return sum(len(value) for value in self._sweep_data.values())

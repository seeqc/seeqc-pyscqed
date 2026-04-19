""""""
import numpy as np

from typing import TypeAlias

from .parameters import ParamCollection


SweepVector: TypeAlias = np.ndarray[np.float64 | np.int64]
ALLOWED_DTYPES = {"int64", "float64"}


class SweepSpec:
    def __init__(self, collection: ParamCollection):
        self._collection = collection
        self._sweep_data: dict[str, SweepVector] = {}

    def add(self, name: str, values: list[float] | SweepVector):
        """Adds a sweep dimension with the specified values."""
        if name in self._sweep_data:
            raise ValueError(f"Parameter \"{name}\" is already in the sweep.")
        if name not in self._collection.getParameterNamesList():
            raise ValueError(f"Parameter \"{name}\" is not in the parameter collection.")
        internal_values = np.array(values)
        if internal_values.ndim != 1:
            raise ValueError(f"Parameter \"{name}\" sweep values must be one-dimensional.")
        print(internal_values.dtype)
        if internal_values.dtype.name not in ALLOWED_DTYPES:
            raise TypeError(f"Parameter \"{name}\" sweep values must be of types {ALLOWED_DTYPES}.")
        self._sweep_data[name] = internal_values

    def get_total_count(self) -> int:
        """Gets the total number of sweep points."""
        return int(np.prod([len(value) for value in self._sweep_data.values()]))

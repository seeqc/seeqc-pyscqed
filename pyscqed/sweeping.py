"""Defines the sweeping API"""
import os

import numpy as np
import sympy as sy

from typing import TypeAlias, Generator

from .parameters import ParamCollection
from .util import pickleRead


SweepValue: TypeAlias = np.float64 | np.int64
SweepPoint: TypeAlias = dict[str, SweepValue]
SweepVector: TypeAlias = np.ndarray[SweepValue]
SweepSubstitution: TypeAlias = dict[sy.Symbol, SweepValue]
ALLOWED_DTYPES = {"int64", "float64"}


class SweepConfig:
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
        if internal_values.dtype.name not in ALLOWED_DTYPES:
            raise TypeError(f"Parameter \"{name}\" sweep values must be of types {ALLOWED_DTYPES}.")
        self._sweep_data[name] = internal_values

    def getTotalCount(self) -> int:
        """Gets the total number of sweep points."""
        return int(np.prod([len(value) for value in self._sweep_data.values()]))

    def getSweptSymbols(self) -> SweepSubstitution:
        """Returns a symbol value mapping for symbols that will be swept by the current sweep. These will be
        substituted during the sweep runtime.
        """
        # Need to ensure that we substitute the parameterised parameter
        not_for_presub = set()
        for name in self._sweep_data:
            not_for_presub.add(name)
            params = self._collection.getParameterisationsInvolving(name)
            for param in params:
                successors = self._collection.getParameterisationSuccessors(param)
                not_for_presub.add(param)
                for k, v in successors.items():
                    not_for_presub |= set(v)
        actual_sub_names = list(not_for_presub)
        return self._collection.getSymbolValues(*actual_sub_names)

    def getStaticSymbols(self) -> SweepSubstitution:
        """Returns a symbol value mapping for symbols that will not be swept. These are static symbols that
        can be substituted before the sweep runtime.
        """
        # For each parameter we want to find it's parametric dependencies, and then exclude those.
        not_for_presub = set()
        for name in self._sweep_data:
            not_for_presub.add(name)
            params = self._collection.getParameterisationsInvolving(name)
            for param in params:
                successors = self._collection.getParameterisationSuccessors(param)
                not_for_presub.add(param)
                for k, v in successors.items():
                    not_for_presub |= set(v)
        
        non_sweep = list(set(self._collection.getParameterNamesList()) - not_for_presub)
        return self._collection.getSymbolValues(*non_sweep)

    def getGenerator(self) -> Generator[SweepPoint, None, None]:
        """Create a generator for sweeping all points."""
        point_count = self.getTotalCount()
        grid = {
            name: values.flatten() for name, values in 
            zip(
                self._sweep_data.keys(),
                np.meshgrid(*self._sweep_data.values(), indexing="ij")
            )
        }
        for index in range(point_count):
            yield {name: values[index] for name, values in grid.items()}


class SweepResult:
    def __init__(self, sweep_config: SweepConfig, data: np.ndarray):
        self.sweep_config = sweep_config
        self.data: np.ndarray = data

    def get(self, independent_variable: str) -> np.ndarray:
        return self.data.T

    @classmethod
    def from_disk_data(cls, sweep_config: SweepConfig, files: list[str | bytes | os.PathLike]) -> "SweepResult":
        data = np.array([pickleRead(file) for file in files])
        return cls(sweep_config, data)

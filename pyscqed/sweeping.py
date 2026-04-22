"""Defines the sweeping API"""
import os

import numpy as np
import sympy as sy

from abc import abstractmethod
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

    def get(self, name: str) -> SweepVector:
        """Retrieves a sweep dimension."""
        return self._sweep_data[name]

    def getDimensionCount(self) -> int:
        """Gets the number of dimensions in the sweep."""
        return len(self._sweep_data)

    def getTotalCount(self) -> int:
        """Gets the total number of sweep points."""
        return int(np.prod([len(value) for value in self._sweep_data.values()]))

    def getSweepShape(self) -> tuple[int]:
        """Gets the shape of the sweep dimensions."""
        return tuple([len(value) for value in self._sweep_data.values()])

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
                np.meshgrid(*self._sweep_data.values(), indexing="ij", copy=False)
            )
        }
        for index in range(point_count):
            yield {name: values[index] for name, values in grid.items()}

    def getSweepAxes(self) -> dict[str, int]:
        """Get a parameter axis mapping"""
        return {param: axis for axis, param in enumerate(self._sweep_data)}


class SweepResult:
    def __init__(self, sweep_config: SweepConfig, data: np.ndarray):
        self.sweep_config = sweep_config
        self.parameter_axes = sweep_config.getSweepAxes()
        self.data: np.ndarray = data

    def get(self, independent_variable: str, static_variables: dict[str, SweepValue] | None = None) -> np.ndarray:
        if static_variables is None:
            static_variables = {}

        # Check there are enough inputs to retrieve a sweep
        sweep_dimensions = self.sweep_config.getDimensionCount()
        if len(static_variables) + 1 != sweep_dimensions:
            raise ValueError("Insufficient independent and static variables to retrieve sweep result data.")

        # Reshape the data
        reshaped_data = self._reshape_data()

        data_shape = list(reshaped_data.shape)
        slices = [slice(None)] * len(data_shape)

        # Construct the slice specification for static variables
        for param, value in static_variables.items():
            sweep_vector = self.sweep_config.get(param)
            value_index = np.argmin(np.abs(sweep_vector - value))
            slices[self.parameter_axes[param]] = value_index
        
        #return new_data[*slices].T
        return self._slice_data(slices, reshaped_data)

    @abstractmethod
    def _reshape_data(self) -> np.ndarray:
        pass

    @abstractmethod
    def _slice_data(self, slices: list[slice], reshaped_data: np.ndarray) -> np.ndarray:
        pass


class SweepResultFromMemory(SweepResult):
    def __init__(self, sweep_config: SweepConfig, data: np.ndarray):
        super().__init__(sweep_config, data)

    def _reshape_data(self) -> np.ndarray:
        data_shape = list(self.data.shape)
        sweep_shape = list(self.sweep_config.getSweepShape())
        # Reshape from flattened data: The inner-most dimensions correspond to outputs of evaluated functions, and thus could
        # be of variable dimensions.
        reshape_spec = sweep_shape + data_shape[1:]
        return self.data.reshape(*reshape_spec)

    def _slice_data(self, slices: list[slice], reshaped_data: np.ndarray) -> np.ndarray:
        return reshaped_data[*slices].T


class SweepResultFromDisk(SweepResult):
    def __init__(self, sweep_config: SweepConfig, files: list[str | bytes | os.PathLike]):
        data = np.array([str(file) for file in files])
        super().__init__(sweep_config, data)

    def _reshape_data(self) -> np.ndarray:
        sweep_shape = list(self.sweep_config.getSweepShape())
        return self.data.reshape(*sweep_shape)

    def _slice_data(self, slices: list[slice], reshaped_data: np.ndarray) -> np.ndarray:
        sliced_data = reshaped_data[*slices]
        
        # Determine the data shape in the files
        # They should all be the same
        index = [0] * len(sliced_data.shape)
        file = sliced_data[*index]
        sample_data = pickleRead(file)
        file_data_shape = list(sample_data.shape)
        
        # Initialize and populated the final data array
        shape_spec = list(sliced_data.shape) + file_data_shape
        loaded_array = np.zeros(shape_spec)
        for index, file in np.ndenumerate(sliced_data):
            loaded_array[index] = pickleRead(file)
        return loaded_array.T

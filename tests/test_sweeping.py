import pytest

from pyscqed.sweep_spec import SweepSpec
from pyscqed.parameters import ParamCollection


def create_parameter_collection(names: list[str], values: list[float]) -> ParamCollection:
    collection = ParamCollection(names)
    for name, value in zip(names, values):
        collection.setParameterValue(name, value)
    return collection


def test_sweep_creation():
    names = ["C1", "L1"]
    values = [1e-15, 1e-9]
    collection = create_parameter_collection(names, values)
    sweep = SweepSpec(collection)

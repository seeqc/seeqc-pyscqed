import pytest

import numpy as np

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

    # 1D sweep
    points = np.linspace(0.0, 1.0, 11)
    sweep.add("C1", points)
    assert sweep.get_total_count() == 11

    # Cannot add the same sweep twice
    with pytest.raises(
        ValueError,
        match="Parameter \"C1\" is already in the sweep."
    ):
        sweep.add("C1", points)

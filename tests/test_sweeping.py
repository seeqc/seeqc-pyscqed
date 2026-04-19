import pytest

import numpy as np
import sympy as sy

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

    # Cannot add a sweep for a parameter that doesn't exist
    points = np.linspace(0.0, 1.0, 11)
    with pytest.raises(
        ValueError,
        match="Parameter \"C2\" is not in the parameter collection."
    ):
        sweep.add("C2", points)

    # 1D sweep
    sweep.add("C1", points)
    assert sweep.getTotalCount() == 11

    # Cannot add the same sweep twice
    with pytest.raises(
        ValueError,
        match="Parameter \"C1\" is already in the sweep."
    ):
        sweep.add("C1", points)

    # Cannot add bad values for the sweep
    with pytest.raises(
        ValueError,
        match="Parameter \"L1\" sweep values must be one-dimensional."
    ):
        sweep.add("L1", np.linspace(1, 6, 6).reshape(2, 3))

    with pytest.raises(
        ValueError,
        match="Parameter \"L1\" sweep values must be one-dimensional."
    ):
        sweep.add("L1", [[1, 2], [2, 3]])

    with pytest.raises(
        TypeError,
        match="Parameter \"L1\" sweep values must be of types*"
    ):
        sweep.add("L1", ["bad"])

    sweep.add("L1", points)
    assert sweep.getTotalCount() == 11**2


def test_get_swept_and_static_symbols():
    names = ["C1", "L1", "I1"]
    values = [1e-15, 1e-9, 1e-6]
    collection = create_parameter_collection(names, values)
    sweep = SweepSpec(collection)

    points = np.linspace(0.0, 1.0, 11)
    sweep.add("C1", points)
    sweep.add("L1", points)

    swept_symbols = sweep.getSweptSymbols()
    static_symbols = sweep.getStaticSymbols()
    assert len(swept_symbols) == 2
    assert sy.Symbol("C_{1}") in swept_symbols
    assert sy.Symbol("L_{1}") in swept_symbols
    assert len(static_symbols) == 1
    assert sy.Symbol("I_{1}") in static_symbols


def test_sweep_point_generator():
    names = ["C", "L", "I"]
    values = [1e-15, 1e-9, 1e-6]
    collection = create_parameter_collection(names, values)
    sweep = SweepSpec(collection)

    # The inner-most sweep is the last
    pts1 = np.linspace(0.0, 1.0, 3)
    pts2 = np.linspace(2.0, 3.0, 3)
    pts3 = np.linspace(4.0, 5.0, 3)
    sweep.add("C", pts1)
    sweep.add("L", pts2)
    sweep.add("I", pts3)

    assert sweep.getTotalCount() == 27

    # The first 3 values for L should be 2.0 and the first 6 for C should be 0.0
    sweep_generator = sweep.getGenerator()
    for index in range(9):
        next_values = sweep_generator.send(None)
        assert next_values["C"] == 0.0
        if index <= 2:
            assert next_values["L"] == 2.0
        elif index > 2 and index <= 5:
            assert next_values["L"] == 2.5
        else:
            assert next_values["L"] == 3.0

    next_values = sweep_generator.send(None)
    assert next_values["C"] == 0.5
    assert next_values["L"] == 2.0
    assert next_values["I"] == 4.0

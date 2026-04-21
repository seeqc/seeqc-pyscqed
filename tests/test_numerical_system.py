import pytest

import numpy as np

from pyscqed.circuit_graph import CircuitGraph
from pyscqed.symbolic_system import SymbolicSystem
from pyscqed.numerical_system import NumericalSystem


def get_single_node_graph() -> CircuitGraph:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")
    return graph


def get_numerical_system(graph: CircuitGraph) -> NumericalSystem:
    # NOTE: The system simulated here is not configured for generating representative physical results,
    # they are only there to check numerical consistency
    symbolic = SymbolicSystem(graph)
    hamil = NumericalSystem(symbolic)
    hamil.configureOperator(1, 40, "charge")
    return hamil


def test_sweep_one_dimension():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH 
    )

    hamil.newSweep()
    hamil.addSweep('C', 20.0, 40.0, 3)
    sweep = hamil.paramSweep(timesweep=True)

    expected_spectrum_sweep = [
        [59.95010688, 45.30354992, 36.57241861],
        [219.58210001, 175.64339355, 149.45048272],
        [379.20831265, 305.97937504, 262.32564643],
        [538.82879099, 436.31151958, 375.19792611],
        [698.44358094, 566.63985224, 488.06733808]
    ]
    x, C, v = hamil.getSweep(sweep, 'C', {})
    assert np.allclose(C, expected_spectrum_sweep, rtol=0, atol=1e-6)

    sweep = hamil.newSweepConfig()
    sweep.add("C", np.linspace(20.0, 40.0, 3))
    result = hamil.runSweep(sweep)
    assert np.allclose(result.get("C"), expected_spectrum_sweep, rtol=0, atol=1e-6)


def test_sweep_two_dimensions():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH 
    )

    hamil.newSweep()
    hamil.addSweep('C', 20.0, 40.0, 3)
    hamil.addSweep('L', 50.0, 60.0, 2)
    sweep = hamil.paramSweep(timesweep=True)

    expected_spectrum_sweep = [
        [59.95010688, 45.30354992, 36.57241861],
        [219.58210001, 175.64339355, 149.45048272],
        [379.20831265, 305.97937504, 262.32564643],
        [538.82879099, 436.31151958, 375.19792611],
        [698.44358094, 566.63985224, 488.06733808]
    ]
    x, C, v = hamil.getSweep(sweep, 'C', {"L": 50.0})
    assert np.allclose(C, expected_spectrum_sweep, rtol=0, atol=1e-6)

    sweep = hamil.newSweepConfig()
    sweep.add("C", np.linspace(20.0, 40.0, 3))
    sweep.add("L", np.linspace(50.0, 60.0, 2))
    result = hamil.runSweep(sweep)

    with pytest.raises(
        ValueError,
        match="Insufficient independent and static variables to retrieve sweep result data."
    ):
        result.get("C")

    assert np.allclose(result.get("C", {"L": 50.0}), expected_spectrum_sweep, rtol=0, atol=1e-6)

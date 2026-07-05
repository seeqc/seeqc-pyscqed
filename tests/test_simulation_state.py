import pytest

import numpy as np

from pyscqed.circuit_graph import CircuitGraph
from pyscqed.symbolic_system import SymbolicSystem
from pyscqed.simulation_state import _SymbolicParts
from pyscqed.numerical_system import NumericalSystem


def get_single_node_graph() -> CircuitGraph:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")
    return graph


def get_symbolic_system() -> SymbolicSystem:
    return SymbolicSystem(get_single_node_graph())


def test_simulation_state_attributes_match_symbolic_system():
    symbolic = get_symbolic_system()
    parts = _SymbolicParts(symbolic)

    assert parts.inverse_capacitance_matrix == symbolic.getInverseCapacitanceMatrix()
    assert parts.inverse_inductance_matrix == symbolic.getInverseInductanceMatrix()
    assert parts.branch_inverse_inductance_matrix == \
        symbolic.getInverseInductanceMatrix(mode="branch")
    assert parts.josephson_vector == symbolic.getJosephsonVector()
    assert parts.phase_slip_vector == symbolic.getPhaseSlipVector()
    assert parts.charge_bias_vector == symbolic.getChargeBiasVector()
    assert parts.branch_charge_bias_vector == \
        symbolic.Rnb * symbolic.getChargeBiasVector()
    assert parts.branch_flux_bias_vector == symbolic.getFluxBiasVector(mode="branch")
    assert parts.branch_flux_bias_matrix == symbolic.getFluxBiasMatrix(mode="branch")
    assert parts.inductive_flux_bias_vector == symbolic.getFluxBiasVectorInd()


def test_simulation_state_rejects_non_symbolic_system():
    with pytest.raises(TypeError):
        _SymbolicParts(get_single_node_graph())

    with pytest.raises(TypeError):
        _SymbolicParts(None)


def test_numerical_system_holds_symbolic_parts():
    hamil = NumericalSystem(get_symbolic_system())
    assert isinstance(hamil._symbolic_parts, _SymbolicParts)


def test_numerical_system_refreshes_symbolic_parts():
    hamil = NumericalSystem(get_symbolic_system())
    old_parts = hamil._symbolic_parts
    hamil.getSymbolicExpressions()
    assert isinstance(hamil._symbolic_parts, _SymbolicParts)
    assert hamil._symbolic_parts is not old_parts


def test_numerical_system_spectrum_unchanged():
    # Guard that the refactor does not alter numerical results
    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 40, "charge")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    hamil.substitute()
    hamil.prepareOperators()
    H = hamil.getHamiltonian()
    energies, _ = hamil.diagonalize(H)
    expected = [59.95010688, 219.58210001, 379.20831265, 538.82879099, 698.44358094]
    assert np.allclose(energies.data, expected, rtol=0, atol=1e-6)

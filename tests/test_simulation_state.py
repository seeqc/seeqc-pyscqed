import pytest

import numpy as np

from pyscqed.circuit_graph import CircuitGraph
from pyscqed.symbolic_system import SymbolicSystem
from pyscqed.simulation_state import _MixedParts, _NumericalParts, _SymbolicParts
from pyscqed.numerical_system import NumericalSystem


def get_single_node_graph() -> CircuitGraph:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")
    return graph


def get_symbolic_system() -> SymbolicSystem:
    return SymbolicSystem(get_single_node_graph())


def get_initialized_symbolic_system() -> SymbolicSystem:
    symbolic = get_symbolic_system()
    symbolic.setParameterValue("C", 20.0)
    symbolic.setParameterValue("I", 40e-3)
    symbolic.setParameterValue("L", 50.0)
    return symbolic


SYMBOLIC_PART_ATTRIBUTES = [
    "inverse_capacitance_matrix",
    "inverse_inductance_matrix",
    "branch_inverse_inductance_matrix",
    "josephson_vector",
    "phase_slip_vector",
    "charge_bias_vector",
    "branch_charge_bias_vector",
    "branch_flux_bias_vector",
    "branch_flux_bias_matrix",
    "inductive_flux_bias_vector",
]


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


def test_mixed_parts_applies_substitutions():
    symbolic = get_initialized_symbolic_system()
    subs = symbolic.getSymbolValuesDict()
    expected = _SymbolicParts(symbolic)
    mixed = _MixedParts(symbolic, subs)
    for name in SYMBOLIC_PART_ATTRIBUTES:
        assert getattr(mixed, name) == getattr(expected, name).subs(subs)


def test_mixed_parts_empty_substitutions_match_symbolic_parts():
    symbolic = get_symbolic_system()
    expected = _SymbolicParts(symbolic)
    mixed = _MixedParts(symbolic, {})
    for name in SYMBOLIC_PART_ATTRIBUTES:
        assert getattr(mixed, name) == getattr(expected, name)


def test_mixed_parts_rejects_non_symbolic_system():
    with pytest.raises(TypeError):
        _MixedParts(get_single_node_graph(), {})


def test_numerical_parts_from_symbolic_parts():
    symbolic = get_initialized_symbolic_system()
    subs = symbolic.getSymbolValuesDict()
    parts = _SymbolicParts(symbolic)
    numeric = _NumericalParts(parts, subs)

    assert np.allclose(
        numeric.inverse_capacitance_matrix,
        np.asarray(parts.inverse_capacitance_matrix.subs(subs), dtype=np.float64)
    )
    assert np.allclose(
        numeric.inverse_inductance_matrix,
        np.asarray(parts.inverse_inductance_matrix.subs(subs), dtype=np.float64)
    )
    assert np.allclose(
        numeric.branch_inverse_inductance_matrix,
        np.asarray(parts.branch_inverse_inductance_matrix.subs(subs), dtype=np.float64)
    )
    assert numeric.josephson_vector.ndim == 1
    assert np.allclose(
        numeric.josephson_vector,
        np.asarray(parts.josephson_vector.subs(subs), dtype=np.float64)[:, 0]
    )
    assert numeric.phase_slip_vector.ndim == 1
    assert np.allclose(
        numeric.phase_slip_vector,
        np.asarray(parts.phase_slip_vector.subs(subs), dtype=np.float64)[:, 0]
    )
    assert np.allclose(
        numeric.charge_bias_vector,
        np.asarray(parts.charge_bias_vector.subs(subs), dtype=np.float64)
    )
    assert np.allclose(
        numeric.branch_flux_bias_matrix,
        np.asarray(parts.branch_flux_bias_matrix.subs(subs), dtype=np.float64)
    )
    assert np.allclose(
        numeric.inductive_flux_bias_vector,
        np.asarray(parts.inductive_flux_bias_vector.subs(subs), dtype=np.float64)
    )

    flux_diag = np.diag(numeric.branch_flux_bias_matrix)
    assert np.allclose(
        numeric.positive_flux_bias_exponentials, np.exp(2j*np.pi*flux_diag))
    assert np.allclose(
        numeric.negative_flux_bias_exponentials, np.exp(-2j*np.pi*flux_diag))


def test_numerical_parts_from_mixed_parts_matches_full_substitution():
    symbolic = get_initialized_symbolic_system()
    subs = symbolic.getSymbolValuesDict()
    full = _NumericalParts(_SymbolicParts(symbolic), subs)

    C = symbolic.getSymbol("C")
    static = {C: subs[C]}
    swept = {k: v for k, v in subs.items() if k != C}
    numeric = _NumericalParts(_MixedParts(symbolic, static), swept)

    assert np.allclose(
        numeric.inverse_capacitance_matrix, full.inverse_capacitance_matrix)
    assert np.allclose(numeric.josephson_vector, full.josephson_vector)
    assert np.allclose(numeric.charge_bias_vector, full.charge_bias_vector)


def test_numerical_parts_rejects_invalid_source():
    symbolic = get_initialized_symbolic_system()
    with pytest.raises(TypeError):
        _NumericalParts(symbolic, {})

    with pytest.raises(TypeError):
        _NumericalParts(None, {})


def test_numerical_system_substitute_populates_numerical_parts():
    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 40, "charge")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    hamil.substitute()
    assert isinstance(hamil._numerical_parts, _NumericalParts)


def test_sweep_does_not_regenerate_charge_basis_operators():
    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 40, "charge")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    old_charge = hamil.circ_operators[1]["charge"]

    sweep = hamil.newSweepConfig()
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    hamil.runSweep(sweep)

    # Charge basis operators do not depend on the swept parameter, so no
    # regeneration is scheduled and the operators are untouched
    assert hamil.circ_operators[1]["charge"] is old_charge


def test_sweep_regenerates_oscillator_basis_operators():
    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 40, "oscillator")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    old_charge = hamil.circ_operators[1]["charge"]

    sweep = hamil.newSweepConfig()
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    hamil.runSweep(sweep)

    # The oscillator impedance depends on the swept parameter, so regeneration
    # is scheduled and the operator vectors must track the new operators
    assert hamil.circ_operators[1]["charge"] is not old_charge

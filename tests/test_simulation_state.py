import pytest

import numpy as np
import qutip as qt

from pyscqed.circuit_graph import CircuitGraph
from pyscqed.symbolic_system import SymbolicSystem
from pyscqed.operators import ChargeBasisOperators, OscillatorBasisOperators
from pyscqed.simulation_state import (
    CircuitOperators,
    SimulationState,
    _MixedParts,
    _NumericalParts,
    _SymbolicParts,
)
from pyscqed.numerical_system import NumericalSystem
from pyscqed.units import Units


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


def test_numerical_system_holds_simulation_state():
    hamil = NumericalSystem(get_symbolic_system())
    assert isinstance(hamil.state, SimulationState)


def test_simulation_state_initializes_containers_in_constructor():
    state = SimulationState(get_symbolic_system())
    assert isinstance(state.circuit_operators, CircuitOperators)
    assert isinstance(state.symbolic_parts, _SymbolicParts)
    assert state.mixed_parts is None
    assert state.numerical_parts is None


def test_simulation_state_rejects_non_symbolic_system():
    with pytest.raises(TypeError):
        SimulationState(get_single_node_graph())

    with pytest.raises(TypeError):
        SimulationState(None)


def test_simulation_state_get_operator():
    state = SimulationState(get_symbolic_system())
    state.circuit_operators.setNodeOperators(1, ChargeBasisOperators(1, 3))
    state.circuit_operators.generateExpandedOperators()
    for kind in ("charge", "flux", "disp", "disp_adj"):
        assert state.getOperator(1, kind) is state.circuit_operators[1][kind]


def test_simulation_state_substitutions():
    symbolic = get_initialized_symbolic_system()
    subs = symbolic.getSymbolValuesDict()
    state = SimulationState(symbolic)
    state.circuit_operators.setNodeOperators(1, ChargeBasisOperators(1, 3))

    # substitute() also generates the expanded node operators
    state.substitute(subs)
    assert isinstance(state.numerical_parts, _NumericalParts)
    assert isinstance(state.getOperator(1, "charge"), qt.Qobj)

    C = symbolic.getSymbol("C")
    state.substituteStatic({C: subs[C]})
    assert isinstance(state.mixed_parts, _MixedParts)

    state.substituteSwept({k: v for k, v in subs.items() if k != C})
    assert isinstance(state.numerical_parts, _NumericalParts)


def test_simulation_state_hilbert_space_size():
    state = SimulationState(get_symbolic_system())
    state.circuit_operators.setNodeOperators(1, ChargeBasisOperators(1, 3))
    assert state.getHilbertSpaceSize() == 7

    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 3, "charge")
    assert hamil.getHilbertSpaceSize() == 7


def test_simulation_state_sparsity():
    state = SimulationState(get_symbolic_system())
    state.circuit_operators.setNodeOperators(1, ChargeBasisOperators(1, 3))
    assert state.sparsity(qt.qeye(7)) == pytest.approx(1 - 1/7)


def test_numerical_system_spectrum_unchanged():
    # Guard that the refactor does not alter numerical results
    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 40, "charge")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
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
    assert isinstance(hamil.state.numerical_parts, _NumericalParts)


def test_sweep_does_not_regenerate_charge_basis_operators():
    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 40, "charge")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    old_charge = hamil.getOperator(1, "charge")

    sweep = hamil.newSweepConfig()
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    hamil.runSweep(sweep)

    # Charge basis operators do not depend on the swept parameter, so no
    # regeneration is scheduled and the operators are untouched
    assert hamil.getOperator(1, "charge") is old_charge
    assert hamil.state.circuit_operators.charge_op_vector[0, 0] is old_charge


def test_sweep_regenerates_oscillator_basis_operators():
    hamil = NumericalSystem(get_symbolic_system())
    hamil.configureOperator(1, 40, "oscillator")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    old_charge = hamil.getOperator(1, "charge")

    sweep = hamil.newSweepConfig()
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    hamil.runSweep(sweep)

    # The oscillator impedance depends on the swept parameter, so regeneration
    # is scheduled and the operator vectors must track the new operators
    assert hamil.getOperator(1, "charge") is not old_charge
    assert hamil.state.circuit_operators.charge_op_vector[0, 0] is \
        hamil.getOperator(1, "charge")


def test_circuit_operators_expansion():
    ops = CircuitOperators([1])
    node_ops = ChargeBasisOperators(1, 3)
    ops.setNodeOperators(1, node_ops)
    ops.generateExpandedOperators()

    assert ops.getNodeOperators(1) is node_ops
    assert set(ops[1].keys()) == {"charge", "flux", "disp", "disp_adj"}
    assert ops[1]["charge"].shape == (7, 7)

    # The operator vectors collect the expanded operators
    assert ops.charge_op_vector.shape == (1, 1)
    assert ops.flux_op_vector.shape == (1, 1)
    assert ops.charge_op_vector[0, 0] is ops[1]["charge"]
    assert ops.flux_op_vector[0, 0] is ops[1]["flux"]


def test_circuit_operators_expansion_multiple_nodes():
    ops = CircuitOperators([1, 2])
    ops.setNodeOperators(1, ChargeBasisOperators(1, 3))
    ops.setNodeOperators(2, ChargeBasisOperators(2, 2))
    ops.generateExpandedOperators()

    # Operators are expanded into the total Hilbert space of dimension 7 * 5
    assert ops[1]["charge"].shape == (35, 35)
    assert ops[2]["charge"].shape == (35, 35)

    # Operators acting on different nodes commute
    keys = ["charge", "flux", "disp", "disp_adj"]
    for key1 in keys:
        for key2 in keys:
            commutator = qt.commutator(ops[1][key1], ops[2][key2])
            assert commutator.norm() == pytest.approx(0.0)

    # The operator vectors collect the expanded operators in node order
    assert ops.charge_op_vector.shape == (2, 1)
    assert ops.flux_op_vector.shape == (2, 1)
    for i, node in enumerate([1, 2]):
        assert ops.charge_op_vector[i, 0] is ops[node]["charge"]
        assert ops.flux_op_vector[i, 0] is ops[node]["flux"]


def test_circuit_operators_rejects_invalid_node_operators():
    ops = CircuitOperators([1])
    with pytest.raises(TypeError):
        ops.setNodeOperators(1, {"truncation": 3})


def test_circuit_operators_regenerates_dependent_nodes():
    symbolic = get_symbolic_system()
    circuit_ops = CircuitOperators(symbolic.nodes)
    circuit_ops.setNodeOperators(
        1, OscillatorBasisOperators(1, 10, symbolic, Units("CQED1"))
    )
    symbolic.setParameterValue("C", 20.0)
    symbolic.setParameterValue("I", 40e-3)
    symbolic.setParameterValue("L", 50.0)
    circuit_ops.generateExpandedOperators()
    old_charge = circuit_ops[1]["charge"]

    # Symbols the operators do not depend on leave them untouched
    circuit_ops.regenerateDependentOperators({symbolic.getSymbol("I")})
    assert circuit_ops[1]["charge"] is old_charge
    assert circuit_ops.charge_op_vector[0, 0] is old_charge

    # Symbols the impedance depends on trigger regeneration
    circuit_ops.regenerateDependentOperators({symbolic.getSymbol("C")})
    assert circuit_ops[1]["charge"] is not old_charge
    assert circuit_ops.charge_op_vector[0, 0] is circuit_ops[1]["charge"]


def test_numerical_system_holds_circuit_operators():
    hamil = NumericalSystem(get_symbolic_system())
    assert isinstance(hamil.state.circuit_operators, CircuitOperators)

    hamil.configureOperator(1, 40, "charge")
    assert isinstance(
        hamil.state.circuit_operators.getNodeOperators(1), ChargeBasisOperators)

    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    for kind in ("charge", "flux", "disp", "disp_adj"):
        assert isinstance(hamil.getOperator(1, kind), qt.Qobj)

import pytest

import sympy as sy

from pyscqed.circuit_graph import CircuitGraph
from pyscqed.symbolic_system import SymbolicSystem
from pyscqed.operators import (
    ChargeBasisOperators,
    NodeOperators,
    OscillatorBasisOperators,
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


def test_node_operators_base_interface():
    ops = NodeOperators(1, 10)
    assert ops.node == 1
    assert ops.truncation == 10
    assert ops.Q is None
    assert ops.P is None
    assert ops.D is None
    assert ops.Ddag is None
    assert ops.dependsOnSymbols({sy.Symbol("x")}) is False

    with pytest.raises(NotImplementedError):
        ops.generate()
    with pytest.raises(NotImplementedError):
        ops.dimension


def test_charge_basis_operators_generate():
    ops = ChargeBasisOperators(1, 40)
    assert isinstance(ops, NodeOperators)
    assert ops.dimension == 2*40 + 1
    assert ops.getIdentity().shape == (81, 81)

    ops.generate()
    for op in (ops.Q, ops.P, ops.D, ops.Ddag):
        assert op.shape == (81, 81)


def test_oscillator_basis_operators_generate():
    symbolic = get_symbolic_system()
    ops = OscillatorBasisOperators(1, 40, symbolic, Units("CQED1"))
    assert isinstance(ops, NodeOperators)
    assert ops.dimension == 40
    assert ops.getIdentity().shape == (40, 40)

    # Construction registers the oscillator parameterisations
    assert "fosc1" in symbolic.getParameterNamesList()
    assert "Zosc1" in symbolic.getParameterNamesList()

    symbolic.setParameterValue("C", 20.0)
    symbolic.setParameterValue("I", 40e-3)
    symbolic.setParameterValue("L", 50.0)

    ops.generate()
    for op in (ops.Q, ops.P, ops.D, ops.Ddag):
        assert op.shape == (40, 40)


def test_oscillator_basis_operators_symbol_dependence():
    symbolic = get_initialized_symbolic_system()
    ops = OscillatorBasisOperators(1, 40, symbolic, Units("CQED1"))

    # The impedance depends on the capacitance and inductance, not the junction
    assert ops.dependsOnSymbols({symbolic.getSymbol("C")})
    assert ops.dependsOnSymbols({symbolic.getSymbol("L")})
    assert not ops.dependsOnSymbols({symbolic.getSymbol("I")})


def test_configure_operator_creates_node_operators():
    hamil = NumericalSystem(get_symbolic_system())

    hamil.configureOperator(1, 40, "charge")
    assert isinstance(hamil.operator_data[1], ChargeBasisOperators)

    hamil.configureOperator(1, 30, "oscillator")
    assert isinstance(hamil.operator_data[1], OscillatorBasisOperators)
    assert hamil.operator_data[1].truncation == 30

    with pytest.raises(Exception, match="basis"):
        hamil.configureOperator(1, 20, "flux")

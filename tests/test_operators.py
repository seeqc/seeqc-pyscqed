import pytest

import numpy as np
import sympy as sy
import qutip as qt

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
    ops = OscillatorBasisOperators(1, 40)
    assert isinstance(ops, NodeOperators)
    assert ops.dimension == 40
    assert ops.getIdentity().shape == (40, 40)

    # Generation requires the symbolic system and units
    with pytest.raises(ValueError):
        ops.generate()

    symbolic.setParameterValue("C", 20.0)
    symbolic.setParameterValue("I", 40e-3)
    symbolic.setParameterValue("L", 50.0)

    ops.generate(symbolic, Units("CQED1"))
    for op in (ops.Q, ops.P, ops.D, ops.Ddag):
        assert op.shape == (40, 40)

    # The first generation registers the oscillator parameterisations and
    # makes their values immediately available
    assert "fosc1" in symbolic.getParameterNamesList()
    assert "Zosc1" in symbolic.getParameterNamesList()
    assert symbolic.getParameterValue("Zosc1") is not None


def test_oscillator_basis_operators_symbol_dependence():
    symbolic = get_initialized_symbolic_system()
    ops = OscillatorBasisOperators(1, 40)

    # Dependencies are unknown until the operators are first generated
    assert not ops.dependsOnSymbols({symbolic.getSymbol("C")})

    ops.generate(symbolic, Units("CQED1"))

    # The impedance depends on the capacitance and inductance, not the junction
    assert ops.dependsOnSymbols({symbolic.getSymbol("C")})
    assert ops.dependsOnSymbols({symbolic.getSymbol("L")})
    assert not ops.dependsOnSymbols({symbolic.getSymbol("I")})


def test_configure_operator_creates_node_operators():
    hamil = NumericalSystem(get_symbolic_system())

    hamil.configureOperator(1, 40, "charge")
    node_ops = hamil.state.circuit_operators.getNodeOperators(1)
    assert isinstance(node_ops, ChargeBasisOperators)

    hamil.configureOperator(1, 30, "oscillator")
    node_ops = hamil.state.circuit_operators.getNodeOperators(1)
    assert isinstance(node_ops, OscillatorBasisOperators)
    assert node_ops.truncation == 30

    with pytest.raises(Exception, match="basis"):
        hamil.configureOperator(1, 20, "flux")


def test_charge_basis_construction():
    trunc = 3
    dim = trunc * 2 + 1
    ops = ChargeBasisOperators(1, trunc)
    ops.generate()

    # Charge operator
    for j in range(dim):
        sj = qt.basis(dim, j).dag()
        for k in range(dim):
            sk = qt.basis(dim, k)
            if j == k:
                # The charge expectation values are integers
                assert sj * ops.Q * sk == pytest.approx(trunc - k)
                continue

            # The off-diagonal elements are zero
            element = sj * ops.Q * sk
            assert element == pytest.approx(0.0)

    # Flux operator
    phi, pstates = ops.P.eigenstates()

    order = np.argsort(phi)
    phi = phi[order]
    pstates = pstates[order]

    # The zero flux state is an equal superposition of the charge states
    assert phi[trunc] == pytest.approx(0.0)
    uniform = sum(qt.basis(dim, k) for k in range(dim)).unit()
    assert abs(uniform.dag() * pstates[trunc]) == pytest.approx(1.0)

    # Displacement operators
    for k in range(dim):
        state = qt.basis(dim, k)
        raised = ops.D * state
        lowered = ops.Ddag * state

        if k == 0:
            # The maximum charge state is annihilated
            assert raised.norm() == pytest.approx(0.0)
        else:
            # D raises the charge by one Cooper pair
            assert qt.basis(dim, k - 1).dag() * raised == pytest.approx(1.0)

        if k == dim - 1:
            # The minimum charge state is annihilated
            assert lowered.norm() == pytest.approx(0.0)
        else:
            # Ddag lowers the charge by one Cooper pair
            assert qt.basis(dim, k + 1).dag() * lowered == pytest.approx(1.0)

    # The normal commutation relation no longer holds, but the result should be non-zero
    assert qt.commutator(ops.Q, ops.P).norm() > 0.0
    assert qt.commutator(ops.Q, ops.Q).norm() == 0.0
    assert qt.commutator(ops.P, ops.P).norm() == 0.0


def test_oscillator_basis_construction():
    trunc = 10
    symbolic = get_symbolic_system()
    ops = OscillatorBasisOperators(1, trunc)
    symbolic.setParameterValue("C", 20.0)
    symbolic.setParameterValue("I", 40e-3)
    symbolic.setParameterValue("L", 50.0)
    ops.generate(symbolic, Units("CQED1"))

    # Charge and flux operators are Hermitian and couple only neighbouring Fock states
    assert ops.Q.isherm
    assert ops.P.isherm
    for j in range(trunc):
        sj = qt.basis(trunc, j).dag()
        for k in range(trunc):
            sk = qt.basis(trunc, k)
            if abs(j - k) != 1:
                assert sj * ops.Q * sk == pytest.approx(0.0)
                assert sj * ops.P * sk == pytest.approx(0.0)

    # The couplings are imaginary (charge) and real (flux), with ladder operator scaling
    q01 = qt.basis(trunc, 0).dag() * ops.Q * qt.basis(trunc, 1)
    p01 = qt.basis(trunc, 0).dag() * ops.P * qt.basis(trunc, 1)
    assert abs(q01) > 0.0
    assert q01.real == pytest.approx(0.0)
    assert abs(p01) > 0.0
    assert p01.imag == pytest.approx(0.0)
    for n in range(trunc - 1):
        sn = qt.basis(trunc, n).dag()
        sn1 = qt.basis(trunc, n + 1)
        assert sn * ops.Q * sn1 == pytest.approx(q01 * np.sqrt(n + 1))
        assert sn * ops.P * sn1 == pytest.approx(p01 * np.sqrt(n + 1))

    # The displacement operators are unitary and commute with the flux operator
    identity = qt.qeye(trunc)
    assert (ops.D * ops.Ddag - identity).norm() == pytest.approx(0.0, abs=1e-9)
    assert qt.commutator(ops.D, ops.P).norm() == pytest.approx(0.0, abs=1e-9)

    # The commutation relation is constant away from the truncation boundary
    commutator = qt.commutator(ops.Q, ops.P)
    c0 = qt.basis(trunc, 0).dag() * commutator * qt.basis(trunc, 0)
    assert abs(c0) > 0.0
    for n in range(trunc - 1):
        sn = qt.basis(trunc, n)
        assert sn.dag() * commutator * sn == pytest.approx(c0)

    # The highest Fock state violates it due to the truncation
    top = qt.basis(trunc, trunc - 1)
    assert top.dag() * commutator * top == pytest.approx(-(trunc - 1) * c0)

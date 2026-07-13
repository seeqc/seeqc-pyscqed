import pytest

import numpy as np
import qutip as qt

from pyscqed.util import mdot, conjugateTranspose


def make_operator_vector(operators: list[qt.Qobj]) -> np.ndarray:
    vector = np.empty((len(operators), 1), dtype=object)
    for i, op in enumerate(operators):
        vector[i, 0] = op
    return vector


def test_conjugate_transpose_daggers_and_transposes():
    a = qt.Qobj([[1.0, 2.0j], [0.0, 3.0]])
    b = qt.Qobj([[0.0, 1.0], [1.0j, 2.0]])
    vector = make_operator_vector([a, b])

    result = conjugateTranspose(vector)

    # Column vector becomes a row vector of adjoints
    assert result.shape == (1, 2)
    assert (result[0, 0] - a.dag()).norm() == pytest.approx(0.0)
    assert (result[0, 1] - b.dag()).norm() == pytest.approx(0.0)


def test_conjugate_transpose_folds_non_hermitian_residue():
    # Operators carrying a deliberate non-Hermitian residue
    residue = 1e-6
    a = qt.Qobj([[1.0, residue*1.0j], [0.0, 2.0]])
    b = qt.Qobj([[0.0, 0.0], [residue*1.0j, 1.0]])
    vector = make_operator_vector([a, b])

    A = np.array([[2.0, 1.0], [1.0, 2.0]])  # real symmetric

    hermitian_form = mdot(conjugateTranspose(vector), A, vector)[0, 0]
    naive_form = mdot(vector.T, A, vector)[0, 0]

    # The conjugate-transpose form folds the residue into a symmetric form and is
    # Hermitian by construction
    assert (hermitian_form - hermitian_form.dag()).norm() == pytest.approx(0.0, abs=1e-12)

    # The plain transpose preserves the residue, so it is not Hermitian
    assert (naive_form - naive_form.dag()).norm() > 1e-9


def test_conjugate_transpose_is_noop_for_hermitian_operators():
    # With Hermitian operators there is no residue to fold: both forms agree
    a = qt.Qobj([[1.0, 1.0j], [-1.0j, 2.0]])
    b = qt.Qobj([[3.0, 0.0], [0.0, 4.0]])
    vector = make_operator_vector([a, b])

    A = np.array([[2.0, 1.0], [1.0, 2.0]])

    hermitian_form = mdot(conjugateTranspose(vector), A, vector)[0, 0]
    naive_form = mdot(vector.T, A, vector)[0, 0]

    assert (hermitian_form - naive_form).norm() == pytest.approx(0.0, abs=1e-12)
    assert (hermitian_form - hermitian_form.dag()).norm() == pytest.approx(0.0, abs=1e-12)

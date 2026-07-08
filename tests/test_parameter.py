import pytest

import numpy as np

from pyscqed.parameters import Param


def test_parameter_creation_rules():
    # Naming rules
    with pytest.raises(TypeError):
        Param(1)
    with pytest.raises(ValueError):
        Param("bad name")
    p = Param("good_name")
    assert p.getValue() is None
    # Value rules
    with pytest.raises(TypeError):
        Param("good_name", value="s")
    for val in [0.5, 1, np.float64(1.2)]:
        p = Param("good_name", value=val)
        assert p.getValue() == val
    # Bounds rules
    with pytest.raises(TypeError):
        Param("good_name", bounds="s")
    with pytest.raises(TypeError):
        Param("good_name", bounds=[1.0, "s"])
    with pytest.raises(TypeError):
        Param("good_name", bounds=["s", 1.0])
    with pytest.raises(ValueError):
        Param("good_name", bounds=[1.0, -1.0])
    p = Param("good_name", bounds=[-0.5, 0.5])
    with pytest.raises(ValueError):
        p.setValue(-0.50001)
    p.setValue(0.2)
    assert p.getValue() == 0.2
    # Unit prefactor rules
    with pytest.raises(TypeError):
        Param("good_name", unit_pref="s")
    with pytest.raises(ValueError):
        Param("good_name", unit_pref=-1.0)
    # Latex names
    assert p.name_latex == "g_\\mathrm{ood_name}"


def test_parameter_functions():
    p = Param("good_name", bounds=[-0.5, 0.5])
    assert p.getBounds() == [-0.5, 0.5]
    with pytest.raises(TypeError):
        p.setBounds("s")
    with pytest.raises(TypeError):
        p.setBounds([1.0, "s"])
    with pytest.raises(TypeError):
        p.setBounds(["s", 1.0])
    with pytest.raises(ValueError):
        p.setBounds([1.0, -1.0])
    p.setBounds([-0.5, 0.5])
    sweep = p.linearSweep(-0.5, 0.5, 2)
    assert all(x == y for x, y in zip(sweep, p.sweep))
    with pytest.raises(TypeError):
        p.linearSweep("s", 1, 2)
    with pytest.raises(TypeError):
        p.linearSweep(1, "s", 2)
    with pytest.raises(TypeError):
        p.linearSweep(0., 1, "s")
    with pytest.raises(TypeError):
        p.linearSweep(0, 1., 2.0)
    with pytest.raises(ValueError):
        p.linearSweep(0, 1.0, 2)
    with pytest.raises(ValueError):
        p.linearSweep(-0.501, 1.0, 2)

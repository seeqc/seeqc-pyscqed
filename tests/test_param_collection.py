import os

import pytest

import numpy as np
import sympy as sy
import graphviz as gv

from pyscqed.parameter_collection import ParamCollection


def check_numerically_equal(Expr1, Expr2, n=100):
    """ Adapted from https://stackoverflow.com/questions/37112738/sympy-comparing-expressions
    """
    # Determine over what range to generate random numbers
    sample_min = -1
    sample_max = 1

    # Regroup all free symbols from both expressions
    free_symbols = set(Expr1.free_symbols) | set(Expr2.free_symbols)

    # Numeric (brute force) equality testing n-times
    for i in range(n):
        your_values = np.random.uniform(sample_min, sample_max, len(free_symbols))
        Expr1_num=Expr1
        Expr2_num=Expr2
        for symbol, number in zip(free_symbols, your_values):
            Expr1_num=Expr1_num.subs(symbol, sy.Float(number))
            Expr2_num=Expr2_num.subs(symbol, sy.Float(number))
        Expr1_num=complex(Expr2_num)
        Expr2_num=complex(Expr2_num)
        if not np.allclose(Expr1_num, Expr2_num, rtol=0, atol=1e-15):
            return False
    return True


def check_symbolically_equal(Expr1, Expr2):
    if (Expr1.equals(Expr2)):
        return True
    return False


NAMES = ["Jc", "L1", "L2", "Long_one"]
VALUES = [1.0, -0.5, -0.25, 1e-9]


def test_paramcollection_creation_rules():
    ParamCollection(NAMES)
    with pytest.raises(TypeError):
        ParamCollection([1])
    ParamCollection([])


def test_getters():
    pc = ParamCollection(NAMES)
    assert all(x == y for x, y in zip(pc.getParameterList(), NAMES))


def test_setters():
    pc = ParamCollection(NAMES)
    assert not pc.allParametersSet()
    set1 = {NAMES[i]: VALUES[i] for i in range(len(NAMES))}
    pc.setParameterValues(set1)
    set2 = []
    for i in range(len(NAMES)):
        set2.append(NAMES[i])
        set2.append(VALUES[i])
    pc.setParameterValues(*set2)
    assert pc.allParametersSet()


def test_parameter_registration_rules():
    pc = ParamCollection(NAMES)

    # addParameter rejects non-string names and invalid symbol overrides
    with pytest.raises(TypeError):
        pc.addParameter(1)
    with pytest.raises(TypeError):
        pc.addParameter("Lx", symbol_override="s")

    # Adding an existing parameter is a no-op that preserves the original symbol
    symbol = pc.getSymbol("L1")
    pc.addParameter("L1")
    assert pc.getSymbol("L1") is symbol

    # addParameterisation rejects unknown parameters and unregistered symbols
    symbols = pc.getSymbolList()
    with pytest.raises(ValueError):
        pc.addParameterisation("unknown", symbols["L1"] + symbols["L2"])
    with pytest.raises(ValueError):
        pc.addParameterisation("Jc", sy.Symbol("unregistered"))


def test_parametric_expressions_symbolic():
    # Setup the param collection
    pc = ParamCollection(NAMES)
    pc.getSymbols("Ltot", "wweird")
    assert {"Ltot", "wweird"} < set(pc.getParameterNamesList())

    # Can create a new parametric expression that depends on existing parameters
    symbols = pc.getSymbolList()
    pc.addParameterisation("Ltot", (symbols["L1"] + symbols["L2"]) * 0.5)
    assert {"Ltot"} == set(pc.getParametricParametersList())
    expected = symbols["L1"]/2 + symbols["L2"]/2
    assert sy.simplify(pc.getParametricExpression("Ltot") - expected) == 0

    # Can create a new parametric expression that makes existing parameters dependent on others
    pc.addParameterisation("Long_one", (10*sy.cos(2*sy.pi*symbols["wweird"]*symbols["Jc"])))
    assert {"Long_one", "Ltot"} == set(pc.getParametricParametersList())

    # Cannot create a circular dependency
    with pytest.raises(AssertionError):
        pc.addParameterisation("wweird", 2 * symbols["Long_one"])
    assert {"Long_one", "Ltot"} == set(pc.getParametricParametersList())

    # Can create a nested parameterisation
    pc.addParameterisation("wweird", sy.sqrt(symbols["Ltot"]))
    assert {"wweird", "Long_one", "Ltot"} == set(pc.getParametricParametersList())
    assert check_numerically_equal(pc.getParametricExpression("Ltot"), expected)

    # By default the nested parameterisations are not expanded
    expected = sy.sqrt(symbols["L1"]/2 + symbols["L2"]/2)
    assert not check_symbolically_equal(pc.getParametricExpression("wweird"), expected)
    assert check_numerically_equal(pc.getParametricExpression("wweird", True), expected)

    # Can get the parameters involved in the parameterisation to depth 1.
    assert set(pc.getParameterisationParameters("Long_one")) == set(['Jc', 'wweird'])

    # Can remove parameterisations
    pc.rmParameterisation("Ltot")
    assert {"wweird", "Long_one"} == set(pc.getParametricParametersList())
    pc.rmParameterisation("Long_one")
    assert {"wweird"} == set(pc.getParametricParametersList())
    expected = symbols["L1"]/2 + symbols["L2"]/2
    with pytest.raises(ValueError):
        pc.getParametricExpression("Ltot")
    with pytest.raises(ValueError):
        pc.getParametricExpression("Long_one")

    # Can write get the parameterisation graph
    obj = pc.drawParameterisationGraph("test1.dot")
    assert type(obj) == gv.Source
    assert os.path.exists("./test1.dot")
    os.remove("./test1.dot")  # Comment to visualize with `dot -Tsvg .\test.dot -o test.svg`

    # Cannot redefine a parameter using the same name
    pc.getSymbols("a")
    symbols = pc.getSymbolList()
    with pytest.raises(AssertionError):
        pc.addParameterisation("L1", symbols["L1"] * symbols["a"])


def test_parametric_expressions_numeric():
    pc = ParamCollection(NAMES)
    pc.getSymbols("Ltot", "wweird")
    symbols = pc.getSymbolList()
    pc.addParameterisation("Ltot", (symbols["L1"] + symbols["L2"]) * 0.5)
    pc.addParameterisation("Long_one", (10*sy.cos(2*sy.pi*symbols["wweird"]*symbols["Jc"])))
    pc.addParameterisation("wweird", sy.sqrt(symbols["Ltot"]))
    # Partial setting of parameters is allowed, parameters that cannot be updated remain None
    pc.setParameterValues({
        "L1": 0.5,
        "L2": 2.0
    })
    values_dict = pc.getParameterValuesDict()
    # Only "Long_one" parameter is missing a value, as it depends on "Jc"
    assert values_dict['Long_one'] is None
    pc.setParameterValues({
        "Jc": 0.1,
        "L1": 1.0,
        "L2": 2.0
    })
    values_dict = pc.getParameterValuesDict()
    L1 = pc.getParameterValue("L1")
    L2 = pc.getParameterValue("L2")
    Jc = pc.getParameterValue("Jc")
    # Does substitution honour the parametric equations?
    assert np.isclose(values_dict['Ltot'], 0.5 * (L1 + L2), rtol=0, atol=1e-15)
    assert np.isclose(values_dict['wweird'], np.sqrt(0.5 * (L1 + L2)), rtol=0, atol=1e-15)
    assert np.isclose(
        values_dict['Long_one'], 10*np.cos(2*np.pi*np.sqrt(0.5 * (L1 + L2))*Jc), rtol=0, atol=1e-14
    )

from typing import Callable

import numpy as np
import sympy as sy
import sympy.utilities as syu

from .numerical_system import NumericalSystem


class ClassicalPotentialBuilder:
    """Used internally to build a numerical classical potential energy getter as a function of all the parameters.
    """
    def __init__(self, hamil: NumericalSystem) -> None:
        self._hamil = hamil
        self._dof_map = {}
        self._critical_currents = self._hamil.getPrefactor('Ej') * self._get_critical_currents()
        self._inverse_inductance_matrix = 0.5 * hamil.getPrefactor('El') * \
            self._hamil.state.numerical_parts.inverse_inductance_matrix
        self._dof_symbol_vector = self._hamil.SS.getFluxVector(mode="branch") + \
                                 self._hamil.SS.getFluxBiasVector(mode="branch")
        self._get_input_format()
        self._symbolic_inductive_energy = self._get_symbolic_inductive_energy()

        # Create numerical functions
        self.jj_func = syu.lambdify(list(self._dof_map.values()), self._dof_symbol_vector)
        self.ind_func = syu.lambdify(list(self._dof_map.values()), self._symbolic_inductive_energy)

    def _get_critical_currents(self) -> np.ndarray:
        subs = self._hamil.SS.getSymbolValuesDict()
        assert all(val is not None for val in subs.values()), "All parameters need to be initialized"
        Jvec = self._hamil.SS.getJosephsonVector().transpose().subs(subs)
        return np.array(Jvec).astype(np.float64)

    def _get_symbolic_inductive_energy(self) -> sy.Expr:
        P = self._hamil.SS.getFluxVector()
        Pe = self._hamil.SS.getFluxBiasVectorInd()
        expr = (P + Pe).transpose() * sy.Matrix(self._inverse_inductance_matrix) * (P + Pe)
        return expr[0, 0]

    def _get_input_format(self) -> None:
        # Get the DoF symbol-node map
        self._dof_map = {"phi%i" % n: sym[0] for n, sym in self._hamil.SS.node_dofs.items() if n > 0}

        # Get the bias symbols
        bias_terms = self._hamil.SS.getFluxBiasVector(mode="branch").free_symbols
        for sym in bias_terms:
            name = self._hamil.SS.getParameterFromSymbol(sym)
            self._dof_map[name] = sym

    def getDefaultInputs(self) -> dict[str, float]:
        return {name: 0.0 for name in self._dof_map}

    def getPotentialFunction(self) -> Callable[[dict[str, float | int | np.ndarray]], float | np.ndarray]:
        def potential(inputs: dict[str, float | int | np.ndarray]) -> float | np.ndarray:
            # Check inputs
            assert isinstance(inputs, dict), "inputs should be a dict instance"
            for name in self._dof_map:
                assert name in inputs, f"expected attribute {name} in inputs"
                assert isinstance(inputs[name], (float, int, np.float64, np.ndarray))

            # Order the arguments correctly
            args = [inputs[name] for name in self._dof_map]

            # Turn the JJ vector into numerical function and get the values
            result = 0.0
            for i, I in enumerate(self._critical_currents[0]):
                result += I*np.cos(2*np.pi*self.jj_func(*args)[i, 0])

            # Calculate the inductive terms
            result += self.ind_func(*args)
            return result

        return potential

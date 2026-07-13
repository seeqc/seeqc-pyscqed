"""Containers and manager for the derived state used by the numerical simulation."""
import numpy as np
import qutip as qt
import sympy as sy

from .operators import NodeOperators
from .symbolic_system import SymbolicSystem
from .units import Units
from . import physical_constants as pc
from . import util


class _SymbolicPartsBase:
    """Base container of the symbolic expressions extracted from a
    :class:`~pyscqed.symbolic_system.SymbolicSystem` that are required to build the numerical
    Hamiltonian. Subclasses define :meth:`_prepare`, applied to each expression as it is
    collected.

    :raises TypeError: if ``symbolic_system`` is not a
        :class:`~pyscqed.symbolic_system.SymbolicSystem` instance.
    """

    def __init__(self, symbolic_system: SymbolicSystem) -> None:
        if not isinstance(symbolic_system, SymbolicSystem):
            raise TypeError(
                "symbolic_system must be a SymbolicSystem instance, got '%s'."
                % type(symbolic_system).__name__
            )

        # Inverse capacitance and inductance matrices
        self.inverse_capacitance_matrix = self._prepare(
            symbolic_system.getInverseCapacitanceMatrix())
        self.inverse_inductance_matrix = self._prepare(
            symbolic_system.getInverseInductanceMatrix())

        # Branch inverse inductance matrix for branch current calculations
        self.branch_inverse_inductance_matrix = self._prepare(
            symbolic_system.getInverseInductanceMatrix(mode="branch"))

        # Symbolic expressions independent of a coupled subsystem
        self.josephson_vector = self._prepare(symbolic_system.getJosephsonVector())
        self.charge_bias_vector = self._prepare(symbolic_system.getChargeBiasVector())
        self.branch_charge_bias_vector = symbolic_system.Rnb * self.charge_bias_vector
        self.branch_flux_bias_vector = self._prepare(
            symbolic_system.getFluxBiasVector(mode="branch"))
        self.branch_flux_bias_matrix = self._prepare(
            symbolic_system.getFluxBiasMatrix(mode="branch"))
        self.inductive_flux_bias_vector = self._prepare(
            symbolic_system.getFluxBiasVectorInd())

    def _prepare(self, expression: sy.Matrix) -> sy.Matrix:
        raise NotImplementedError


class _SymbolicParts(_SymbolicPartsBase):
    """Unmodified symbolic expressions extracted from a
    :class:`~pyscqed.symbolic_system.SymbolicSystem`. """

    def _prepare(self, expression: sy.Matrix) -> sy.Matrix:
        return expression


class _MixedParts(_SymbolicPartsBase):
    """Symbolic expressions with the static parameters of a sweep substituted, leaving the
    swept symbols free. """

    def __init__(self, symbolic_system: SymbolicSystem, substitutions: dict[sy.Symbol, float]) -> None:
        self._substitutions = substitutions
        super().__init__(symbolic_system)

    def _prepare(self, expression: sy.Matrix) -> sy.Matrix:
        return expression.subs(self._substitutions)


class _NumericalParts:
    """Numerical arrays required to build the Hamiltonian, obtained by substituting the
    remaining free symbols of a parts container, including the exponentiated bias terms.

    :raises TypeError: if ``parts`` is not a :class:`_SymbolicParts` or :class:`_MixedParts`
        instance.
    """

    def __init__(self, parts: _SymbolicPartsBase, substitutions: dict[sy.Symbol, float]) -> None:
        if not isinstance(parts, _SymbolicPartsBase):
            raise TypeError(
                "parts must be a _SymbolicParts or _MixedParts instance, got '%s'."
                % type(parts).__name__
            )

        def to_array(expression: sy.Matrix) -> np.ndarray:
            return np.asarray(expression.subs(substitutions), dtype=np.float64)

        self.inverse_capacitance_matrix = to_array(parts.inverse_capacitance_matrix)
        self.inverse_inductance_matrix = to_array(parts.inverse_inductance_matrix)
        self.branch_inverse_inductance_matrix = to_array(
            parts.branch_inverse_inductance_matrix)
        self.josephson_vector = to_array(parts.josephson_vector)[:, 0]
        self.charge_bias_vector = to_array(parts.charge_bias_vector) # x 2e
        self.branch_flux_bias_matrix = to_array(parts.branch_flux_bias_matrix)
        self.inductive_flux_bias_vector = to_array(parts.inductive_flux_bias_vector)

        # Exponentiated flux biases
        flux_biases = np.diag(self.branch_flux_bias_matrix)
        self.positive_flux_bias_exponentials = list(np.exp(2j*np.pi*flux_biases))
        self.negative_flux_bias_exponentials = list(np.exp(-2j*np.pi*flux_biases))


class CircuitOperators:
    """Manages the node operator generators and their expansion into the total circuit
    Hilbert space.

    :param node_list: the circuit nodes in matrix ordering.
    """

    def __init__(self, node_list: list[int]) -> None:
        self._node_list = list(node_list)
        self._operator_data = {}
        self._circ_operators = {}
        self.charge_op_vector = None
        self.flux_op_vector = None
        self.generation = 0

    def __getitem__(self, node: int) -> dict[str, qt.Qobj]:
        """ Returns the expanded operator dictionary of the given node. """
        return self._circ_operators[node]

    def __contains__(self, node: int) -> bool:
        """ Returns whether the given node has an assigned operator generator. """
        return node in self._operator_data

    def getNodeOperators(self, node: int) -> NodeOperators:
        """ Returns the operator generator of the given node. """
        return self._operator_data[node]

    def getHilbertSpaceSize(self) -> int:
        """ Returns the total Hilbert space size considering the operator truncations of
        all nodes. """
        ret = 1
        for ops in self._operator_data.values():
            ret *= ops.dimension
        return ret

    def setNodeOperators(self, node: int, node_operators: NodeOperators) -> None:
        """ Assigns the operator generator for the given node.

        :raises TypeError: if ``node_operators`` is not a
            :class:`~pyscqed.operators.NodeOperators` instance.
        """
        if not isinstance(node_operators, NodeOperators):
            raise TypeError(
                "node_operators must be a NodeOperators instance, got '%s'."
                % type(node_operators).__name__
            )
        self._operator_data[node] = node_operators

    def generateExpandedOperators(
        self,
        nodes: list[int] | None = None,
        symbolic_system: SymbolicSystem | None = None,
        units: Units | None = None
    ) -> None:
        """ Generates the operators of each node (all if ``nodes`` is None) and expands them
        into the total Hilbert space. The symbolic system and units are passed through to
        the operator generators that require them. """
        # Generate the Hilbert space expanders
        Ilist = [self._operator_data[node].getIdentity() for node in self._node_list]

        self.charge_op_vector = np.empty((len(self._node_list), 1), dtype=object)
        self.flux_op_vector = np.empty((len(self._node_list), 1), dtype=object)
        for i, node in enumerate(self._node_list):
            # Regenerate the nodes in the list, if provided; skipped nodes keep their
            # existing expanded operators
            if nodes is None or node in nodes:
                ops = self._operator_data[node]
                ops.generate(symbolic_system, units)

                Olist = list(Ilist)
                op_dict = {}
                Olist[i] = ops.Q
                op_dict["charge"] = qt.tensor(Olist)
                Olist[i] = ops.P
                op_dict["flux"] = qt.tensor(Olist)
                Olist[i] = ops.D
                op_dict["disp"] = qt.tensor(Olist)
                Olist[i] = ops.Ddag
                op_dict["disp_adj"] = qt.tensor(Olist)
                self._circ_operators[node] = op_dict

            # Collect the operator vectors in node order
            self.charge_op_vector[i, 0] = self._circ_operators[node]["charge"]
            self.flux_op_vector[i, 0] = self._circ_operators[node]["flux"]
        self.generation += 1

    def generateExpandedShiftingUnitaries(
        self,
        charge_bias_vector: np.ndarray,
        flux_bias_vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        Ilist = [self._operator_data[node].getIdentity() for node in self._node_list]
        vector1 = np.empty((len(self._node_list), 1), dtype=object)
        vector2 = np.empty((len(self._node_list), 1), dtype=object)
        for i, node in enumerate(self._node_list):
            ops = self._operator_data[node]
            flux_value = flux_bias_vector[i, 0]
            U = (1j * (ops.Q * flux_value * 2 * np.pi + ops.P * charge_bias_vector[i, 0])).expm()
            Olist = list(Ilist)
            Olist[i] = U
            vector1[i, 0] = qt.tensor(Olist)
            Olist[i] = U.dag()
            vector2[i, 0] = qt.tensor(Olist)
        return vector1, vector2

    def regenerateDependentOperators(
        self,
        symbols: dict[sy.Symbol, float],
        symbolic_system: SymbolicSystem | None = None,
        units: Units | None = None
    ) -> None:
        """ Regenerates the expanded operators of the nodes whose operators depend on any of
        the given symbols. """
        symbols = set(symbols)
        nodes = [
            node for node, ops in self._operator_data.items()
            if ops.dependsOnSymbols(symbols)
        ]
        if nodes:
            self.generateExpandedOperators(nodes, symbolic_system, units)


class SimulationState:
    """Manages the derived state used by a numerical simulation: the node operators
    expanded into the circuit Hilbert space, and the symbolic, partially substituted
    and fully numerical Hamiltonian parts.

    :param units: the unit system passed to the node operator generators.
    :raises TypeError: if ``symbolic_system`` is not a
        :class:`~pyscqed.symbolic_system.SymbolicSystem` instance.
    """

    def __init__(self, symbolic_system: SymbolicSystem, units: Units = Units("CQED1")) -> None:
        if not isinstance(symbolic_system, SymbolicSystem):
            raise TypeError(
                "symbolic_system must be a SymbolicSystem instance, got '%s'."
                % type(symbolic_system).__name__
            )
        self._symbolic_system = symbolic_system
        self._units = units
        self.circuit_operators = CircuitOperators(symbolic_system.nodes)
        self.symbolic_parts = _SymbolicParts(symbolic_system)
        self.mixed_parts: _MixedParts | None = None
        self.numerical_parts: _NumericalParts | None = None
        self._shifting_cache = None

    def getOperator(self, node: int, kind: str) -> qt.Qobj:
        """ Returns the operator ``kind`` of the given node, expanded into the circuit
        Hilbert space.

        :param kind: one of ``"charge"``, ``"flux"``, ``"disp"`` or ``"disp_adj"``.
        """
        return self.circuit_operators[node][kind]

    def _getShiftingUnitaries(self) -> tuple[np.ndarray, np.ndarray]:
        # Regenerate the shifting unitaries only when the charge or inductive flux bias
        # vectors change value, or when the node operators they are built from have been
        # regenerated (tracked by the operator generation counter).
        charge_bias = self.numerical_parts.charge_bias_vector
        flux_bias = self.numerical_parts.inductive_flux_bias_vector
        generation = self.circuit_operators.generation
        if self._shifting_cache is not None:
            cached_generation, cached_charge, cached_flux, U, Udag = self._shifting_cache
            if (cached_generation == generation
                    and np.array_equal(cached_charge, charge_bias)
                    and np.array_equal(cached_flux, flux_bias)):
                return U, Udag
        U, Udag = self.circuit_operators.generateExpandedShiftingUnitaries(
            charge_bias, flux_bias
        )
        self._shifting_cache = (
            generation, charge_bias.copy(), flux_bias.copy(), U, Udag
        )
        return U, Udag

    def getBiasedChargeOperatorVector(self) -> np.ndarray:
        """ Returns the node charge operator vector including the charge bias offsets. """
        U, Udag = self._getShiftingUnitaries()
        Q = U * self.circuit_operators.charge_op_vector * Udag
        return Q

    def getBiasedFluxOperatorVector(self) -> np.ndarray:
        """ Returns the node flux operator vector including the inductive flux bias
        offsets. """
        U, Udag = self._getShiftingUnitaries()
        P = U * self.circuit_operators.flux_op_vector * Udag
        return P

    def getInverseCapacitanceMatrix(self) -> np.ndarray:
        """ Returns the numerical inverse capacitance matrix. """
        return self.numerical_parts.inverse_capacitance_matrix

    def getInverseInductanceMatrix(self) -> np.ndarray:
        """ Returns the numerical inverse inductance matrix. """
        return self.numerical_parts.inverse_inductance_matrix

    def getBranchInverseInductanceMatrix(self) -> np.ndarray:
        """ Returns the numerical branch inverse inductance matrix used for branch
        current calculations. """
        return self.numerical_parts.branch_inverse_inductance_matrix

    def getJosephsonVector(self) -> np.ndarray:
        """ Returns the numerical Josephson energy vector. """
        return self.numerical_parts.josephson_vector

    def getPositiveFluxBiasExponentials(self) -> list[np.complex128]:
        """ Returns the positively signed exponentiated branch flux bias terms. """
        return self.numerical_parts.positive_flux_bias_exponentials

    def getNegativeFluxBiasExponentials(self) -> list[np.complex128]:
        """ Returns the negatively signed exponentiated branch flux bias terms. """
        return self.numerical_parts.negative_flux_bias_exponentials

    def getHilbertSpaceSize(self) -> int:
        """ Returns the total Hilbert space size considering all currently defined
        operator truncations. """
        return self.circuit_operators.getHilbertSpaceSize()

    def sparsity(self, op: qt.Qobj) -> float:
        """ Returns the sparsity of the given operator relative to the total Hilbert
        space size. """
        return 1 - op.to("CSR").data.as_scipy().nnz/self.getHilbertSpaceSize()**2

    def getCommutator(self, Q: qt.Qobj, P: qt.Qobj, basis: str = "charge") -> qt.Qobj | None:
        """ Returns the corrected commutator of the given conjugate operator pair.

        :param basis: the basis representation of the operators, ``"charge"`` or
            ``"oscillator"``.
        """
        if basis == "charge": # Need to figure out the correction for this case
            dims = 2*pc.e*pc.phi0/pc.hbar
            return qt.commutator(P, Q)*dims
        elif basis == "oscillator":

            # Get highest number operator eigenvalue and associated eigenstate
            enum = Q.shape[0]
            vnum = qt.basis(enum, enum-1)

            # Create the correction matrix
            mat = 1 - (enum)*vnum*vnum.dag()

            # Invert it
            corrmat = qt.Qobj(np.linalg.inv(mat.data.to_array()))

            # Get the correct commutator
            return qt.commutator(P, Q)*corrmat

    def setNodeOperators(self, node: int, node_operators: NodeOperators) -> None:
        """ Assigns the operator generator for the given node.

        An identical reconfiguration is skipped. A differing reconfiguration made after a
        substitution forces a new :meth:`substitute` call so the expanded operators and
        numerical parts remain consistent.
        """
        if node in self.circuit_operators:
            existing = self.circuit_operators.getNodeOperators(node)
            if existing.sameConfiguration(node_operators):
                return
            self.circuit_operators.setNodeOperators(node, node_operators)
            if self.numerical_parts is not None:
                self.substitute(self._symbolic_system.getSymbolValuesDict())
        else:
            self.circuit_operators.setNodeOperators(node, node_operators)

    def substitute(self, substitutions: dict[sy.Symbol, float]) -> None:
        """ Substitutes all parameter values into the symbolic expressions, populating
        :attr:`numerical_parts`, and generates the expanded node operators. """
        self.numerical_parts = _NumericalParts(self.symbolic_parts, substitutions)
        self.circuit_operators.generateExpandedOperators(
            symbolic_system=self._symbolic_system, units=self._units)

    def substituteStatic(self, substitutions: dict[sy.Symbol, float]) -> None:
        """ Substitutes the static parameters of a sweep, leaving the swept symbols
        free and populating :attr:`mixed_parts`. """
        self.mixed_parts = _MixedParts(self._symbolic_system, substitutions)

    def substituteSwept(self, substitutions: dict[sy.Symbol, float]) -> None:
        """ Substitutes the swept parameter values at the current sweep point,
        populating :attr:`numerical_parts` and regenerating the operators that depend
        on the swept symbols. """
        self.numerical_parts = _NumericalParts(self.mixed_parts, substitutions)
        self.circuit_operators.regenerateDependentOperators(
            substitutions, self._symbolic_system, self._units)

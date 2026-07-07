"""Internal containers holding derived state used by the numerical simulation."""
import numpy as np
import qutip as qt

from .operators import NodeOperators
from .symbolic_system import SymbolicSystem


class _SymbolicPartsBase:
    """Base container of the symbolic expressions extracted from a
    :class:`~pyscqed.symbolic_system.SymbolicSystem` that are required to build the numerical
    Hamiltonian. Subclasses define :meth:`_prepare`, applied to each expression as it is
    collected.

    :raises TypeError: if ``symbolic_system`` is not a
        :class:`~pyscqed.symbolic_system.SymbolicSystem` instance.
    """

    def __init__(self, symbolic_system: SymbolicSystem):
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
        self.phase_slip_vector = self._prepare(symbolic_system.getPhaseSlipVector())
        self.charge_bias_vector = self._prepare(symbolic_system.getChargeBiasVector())
        self.branch_charge_bias_vector = symbolic_system.Rnb * self.charge_bias_vector
        self.branch_flux_bias_vector = self._prepare(
            symbolic_system.getFluxBiasVector(mode="branch"))
        self.branch_flux_bias_matrix = self._prepare(
            symbolic_system.getFluxBiasMatrix(mode="branch"))
        self.inductive_flux_bias_vector = self._prepare(
            symbolic_system.getFluxBiasVectorInd())

    def _prepare(self, expression):
        raise NotImplementedError


class _SymbolicParts(_SymbolicPartsBase):
    """Unmodified symbolic expressions extracted from a
    :class:`~pyscqed.symbolic_system.SymbolicSystem`. """

    def _prepare(self, expression):
        return expression


class _MixedParts(_SymbolicPartsBase):
    """Symbolic expressions with the static parameters of a sweep substituted, leaving the
    swept symbols free. """

    def __init__(self, symbolic_system: SymbolicSystem, substitutions):
        self._substitutions = substitutions
        super().__init__(symbolic_system)

    def _prepare(self, expression):
        return expression.subs(self._substitutions)


class _NumericalParts:
    """Numerical arrays required to build the Hamiltonian, obtained by substituting the
    remaining free symbols of a parts container, including the exponentiated bias terms.

    :raises TypeError: if ``parts`` is not a :class:`_SymbolicParts` or :class:`_MixedParts`
        instance.
    """

    def __init__(self, parts: _SymbolicPartsBase, substitutions):
        if not isinstance(parts, _SymbolicPartsBase):
            raise TypeError(
                "parts must be a _SymbolicParts or _MixedParts instance, got '%s'."
                % type(parts).__name__
            )

        def to_array(expression):
            return np.asarray(expression.subs(substitutions), dtype=np.float64)

        self.inverse_capacitance_matrix = to_array(parts.inverse_capacitance_matrix)
        self.inverse_inductance_matrix = to_array(parts.inverse_inductance_matrix)
        self.branch_inverse_inductance_matrix = to_array(
            parts.branch_inverse_inductance_matrix)
        self.josephson_vector = to_array(parts.josephson_vector)[:, 0]
        self.phase_slip_vector = to_array(parts.phase_slip_vector)[:, 0]
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

    def __init__(self, node_list):
        self._node_list = list(node_list)
        self.operator_data = {}
        self.circ_operators = {}

    def __getitem__(self, node):
        """ Returns the expanded operator dictionary of the given node. """
        return self.circ_operators[node]

    def setNodeOperators(self, node, node_operators: NodeOperators):
        """ Assigns the operator generator for the given node.

        :raises TypeError: if ``node_operators`` is not a
            :class:`~pyscqed.operators.NodeOperators` instance.
        """
        if not isinstance(node_operators, NodeOperators):
            raise TypeError(
                "node_operators must be a NodeOperators instance, got '%s'."
                % type(node_operators).__name__
            )
        self.operator_data[node] = node_operators

    def generateExpandedOperators(self, nodes=None):
        """ Generates the operators of each node (all if ``nodes`` is None) and expands them
        into the total Hilbert space. """
        # Generate the Hilbert space expanders
        Ilist = [self.operator_data[node].getIdentity() for node in self._node_list]

        for i, node in enumerate(self._node_list):
            # Ignore nodes that are not in the list, if provided
            if nodes is not None and node not in nodes:
                continue

            ops = self.operator_data[node]
            ops.generate()

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
            self.circ_operators[node] = op_dict

    def regenerateDependentOperators(self, symbols):
        """ Regenerates the expanded operators of the nodes whose operators depend on any of
        the given symbols. """
        symbols = set(symbols)
        nodes = [
            node for node, ops in self.operator_data.items()
            if ops.dependsOnSymbols(symbols)
        ]
        if nodes:
            self.generateExpandedOperators(nodes)

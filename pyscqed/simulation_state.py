"""Internal containers holding derived state used by the numerical simulation."""
import numpy as np

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

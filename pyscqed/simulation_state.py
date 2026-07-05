"""Internal containers holding derived state used by the numerical simulation."""
from .symbolic_system import SymbolicSystem


class _SymbolicParts:
    """Symbolic expressions extracted from a :class:`~pyscqed.symbolic_system.SymbolicSystem`
    that are required to build the numerical Hamiltonian.

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
        self.inverse_capacitance_matrix = symbolic_system.getInverseCapacitanceMatrix()
        self.inverse_inductance_matrix = symbolic_system.getInverseInductanceMatrix()

        # Branch inverse inductance matrix for branch current calculations
        self.branch_inverse_inductance_matrix = \
            symbolic_system.getInverseInductanceMatrix(mode="branch")

        # Symbolic expressions independent of a coupled subsystem
        self.josephson_vector = symbolic_system.getJosephsonVector()
        self.phase_slip_vector = symbolic_system.getPhaseSlipVector()
        self.charge_bias_vector = symbolic_system.getChargeBiasVector()
        self.branch_charge_bias_vector = symbolic_system.Rnb * self.charge_bias_vector
        self.branch_flux_bias_vector = symbolic_system.getFluxBiasVector(mode="branch")
        self.branch_flux_bias_matrix = symbolic_system.getFluxBiasMatrix(mode="branch")
        self.inductive_flux_bias_vector = symbolic_system.getFluxBiasVectorInd()

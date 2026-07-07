"""Operator construction for the circuit degrees of freedom."""
import numpy as np
import qutip as qt
import sympy as sy

from . import physical_constants as pc
from .symbolic_system import SymbolicSystem
from .units import Units

# Local override for the Qobj tolerance. The global settings appear to not work
_qobj_atol = 1e-12


class NodeOperators:
    """Base container that constructs and holds the operators of a single circuit node degree
    of freedom. Subclasses implement the basis-specific generation of the charge (``Q``),
    flux (``P``) and Josephson displacement (``D``, ``Ddag``) operators.
    """

    def __init__(self, node: int, truncation: int) -> None:
        self.node = node
        self.truncation = truncation
        self.Q = None
        self.P = None
        self.D = None
        self.Ddag = None

    @property
    def dimension(self) -> int:
        """ The Hilbert space dimension of this degree of freedom. """
        raise NotImplementedError

    def getIdentity(self) -> qt.Qobj:
        """ Returns the identity operator used to expand into the total Hilbert space. """
        return qt.qeye(self.dimension)

    def generate(self, symbolic_system: SymbolicSystem | None = None, units: Units | None = None) -> None:
        """ Populates ``Q``, ``P``, ``D`` and ``Ddag`` from the current parameter values.

        :param symbolic_system: required by bases whose operators depend on the circuit.
        :param units: required by bases whose operators depend on the unit system.
        """
        raise NotImplementedError

    def dependsOnSymbols(self, symbols: set[sy.Symbol]) -> bool:
        """ Returns whether the operators need regenerating when any of the given symbols
        change value. """
        return False

    def sameConfiguration(self, other: "NodeOperators") -> bool:
        """ Returns whether ``other`` is configured to generate the same operators as
        this instance. """
        return type(other) is type(self) and other.truncation == self.truncation


class ChargeBasisOperators(NodeOperators):
    """Charge basis operators for a single circuit node degree of freedom."""

    @property
    def dimension(self) -> int:
        return 2*self.truncation + 1

    def generate(self, symbolic_system: SymbolicSystem | None = None, units: Units | None = None) -> None:
        trunc = self.truncation

        # For IJ modes the charge states are the eigenvectors of the following matrix
        _, s = (-qt.num(2*trunc + 1, trunc)).eigenstates()

        # Construct the flux states
        phik_list = []
        phik = None
        qm = [float(i)-trunc for i in range(2*trunc + 1)]
        for k in qm:
            phik = qt.basis(2*trunc + 1, 0) * 0
            for j, qi in enumerate(qm):
                phik += np.sqrt(1/(2*trunc + 1)) * np.exp(2j*np.pi*k*qi/(2*trunc + 1))*s[j]
            phik_list.append(phik)

        # From this build the flux operator
        phik_eigvals = [(float(i)-trunc)/(2*trunc + 1) for i in range(2*trunc + 1)]
        P = qt.qeye(2*trunc + 1) - qt.qeye(2*trunc + 1)
        for i, phik in enumerate(phik_list):
            P += phik_eigvals[i]*phik*phik.dag()

        # Get a simple charge number operator
        Q = -qt.num(2*trunc + 1)+float(trunc)

        # Generate Josephson displacement operators by first diagonalising the flux operator
        E, V = np.linalg.eigh(P.data.to_array())

        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))

        # Exponentiate the diagonal matrix
        D = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - \
            qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()

        self.Q = Q.to("CSR").tidyup(_qobj_atol)
        self.P = P.to("CSR").tidyup(_qobj_atol)
        self.D = D.to("CSR").tidyup(_qobj_atol)
        self.Ddag = D.dag().to("CSR").tidyup(_qobj_atol)


class OscillatorBasisOperators(NodeOperators):
    """Harmonic oscillator basis operators for a single circuit node degree of freedom.

    The first generation derives the symbolic oscillator ``frequency`` and ``impedance``
    from the circuit and registers the associated ``fosc<node>`` and ``Zosc<node>``
    parameterisations on the symbolic system.
    """

    def __init__(self, node: int, truncation: int) -> None:
        super().__init__(node, truncation)
        self.frequency = None
        self.impedance = None

    @property
    def dimension(self) -> int:
        return self.truncation

    def dependsOnSymbols(self, symbols: set[sy.Symbol]) -> bool:
        if self.impedance is None:
            return False
        return not self.impedance.free_symbols.isdisjoint(symbols)

    def generate(self, symbolic_system: SymbolicSystem | None = None, units: Units | None = None) -> None:
        if symbolic_system is None or units is None:
            raise ValueError(
                "The symbolic system and units are required to generate oscillator "
                "basis operators."
            )
        if self.impedance is None:
            self._derive_oscillator_parameters(symbolic_system)

        trunc = self.truncation

        # Get the impedance of the mode
        osc_impedance = symbolic_system.getParameterValue("Zosc%i" % self.node) * \
            units.getPrefactor("Impe")

        # Get the prefactor that results from transformation (the charge increment prefactor)
        a = symbolic_system.cooper_disp[self.node]

        # Using oscillator basis
        Q = 1j*np.sqrt(1/(2*osc_impedance))*(qt.create(trunc) - qt.destroy(trunc))*units.getPrefactor("ChgOsc")
        P = np.sqrt(osc_impedance/2)*(qt.create(trunc) + qt.destroy(trunc))*units.getPrefactor("FlxOsc")
        Pp = a*2*np.pi/pc.phi0*np.sqrt(pc.hbar)*P/units.getPrefactor("FlxOsc")

        # Generate Josephson displacement operators by first diagonalising the flux operator
        E, V = np.linalg.eigh(Pp.data.to_array())

        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))

        # Exponentiate the diagonal matrix
        D = U*qt.Qobj(np.diag(np.exp(1j*E)))*Uinv

        self.Q = Q.to("CSR").tidyup(_qobj_atol)
        self.P = P.to("CSR").tidyup(_qobj_atol)
        self.D = D.to("CSR").tidyup(_qobj_atol)
        self.Ddag = D.dag().to("CSR").tidyup(_qobj_atol)

    def _derive_oscillator_parameters(self, symbolic_system: SymbolicSystem) -> None:
        # Derive the oscillator parameters from the circuit
        index = symbolic_system.nodes.index(self.node)
        Linv = symbolic_system.getInverseInductanceMatrix()
        Cinv = symbolic_system.getInverseCapacitanceMatrix()
        self.frequency = sy.sqrt(Linv[index, index]*Cinv[index, index])
        self.impedance = sy.sqrt(Cinv[index, index]/Linv[index, index])

        # Register the oscillator parameterisations and update their values
        freq = "fosc%i" % self.node
        symbolic_system.addParameter(freq)
        symbolic_system.addParameterisation(freq, self.frequency)
        impe = "Zosc%i" % self.node
        symbolic_system.addParameter(impe)
        symbolic_system.addParameterisation(impe, self.impedance)
        symbolic_system.updateParameterisations()

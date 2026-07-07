import qutip as qt
import numpy as np
import sympy as sy
import sympy.utilities as syu
import scipy as sc
import networkx as nx
import progress.bar
import time

from .dataspec import TempData
from .symbolic_system import SymbolicSystem
from .operators import ChargeBasisOperators, OscillatorBasisOperators, _qobj_atol
from .simulation_state import SimulationState
from .sweeping import SweepConfig, SweepResult, SweepResultFromDisk, SweepResultFromMemory
from .evaluation_graph import EvaluationGraph
from .result import EigenvalueResult, EigenvectorResult, FunctionResult
from .units import Units
from . import util
from . import units


class HamiltonianSpectrum(EvaluationGraph):
    def __init__(self, nsys: "NumericalSystem"):
        super().__init__()
        self.addNode("Hamiltonian", fn=nsys.getHamiltonian, outputs=["qobj"])
        self.addNode("Spectrum", fn=nsys.diagonalize, outputs=["E"])
        self.addDependency("Hamiltonian", "Spectrum")


class SingleResonatorInteraction(EvaluationGraph):
    def __init__(self, nsys: "NumericalSystem"):
        super().__init__()
        self.addNode("Hamiltonian", fn=nsys.getHamiltonian, outputs=["qobj"])
        self.addNode("Spectrum", fn=nsys.diagonalize, outputs=["energies", "vectors"])
        self.addNode("Resonator", fn=nsys.getResonatorResponse, outputs=["rwa_energies"])
        self.addDependency("Hamiltonian", "Spectrum")
        self.addDependency("Spectrum", "Resonator")


class NumericalSystem(TempData):
    
    ## Mode types
    __mode_types = [
        "osc", # DoFs have capacitive and inductive parts only
        "jos", # DoFs have capacitive, inductive and Josephson parts
        "isl", # DoFs have capacitive parts only
        "leg"  # DoFs have inductive parts only
    ]
    
    ## Initialise a Hamiltonian using a circuit specification
    def __init__(self, symbolic_system: SymbolicSystem, unit=Units("CQED1")):
        
        # Initialise the temporary data manager
        super().__init__()
        self.newSession(id(self))
        self.__use_temp = False
        
        # Assign the circuit
        self.SS = symbolic_system

        # Manager for the derived circuit state
        self.state = SimulationState(symbolic_system, unit)

        # Set the unit system
        self.units = unit
        self._set_parameter_units()

        # Load default diagonaliser configuration
        self.setDiagConfig()

    def getNodeList(self):
        return self.SS.nodes
    
    def getNodeIndex(self, node):
        return self.SS.nodes.index(node)
    
    def getEdgeList(self):
        return self.SS.edges
    
    def getEdgeIndex(self, edge):
        return self.SS.edges.index(edge)
    
    def getCircuitGraph(self):
        return self.SS.CG
    
    def getSymbolicSystem(self):
        return self.SS

    def getOperator(self, node, kind):
        """ Returns the operator ``kind`` of the given node, expanded into the circuit
        Hilbert space.

        :param kind: one of ``"charge"``, ``"flux"``, ``"disp"`` or ``"disp_adj"``.
        """
        return self.state.getOperator(node, kind)

    def getHilbertSpaceSize(self):
        """ Returns the Hilbert space size considering all currently defined operator
        truncations. """
        return self.state.getHilbertSpaceSize()

    def sparsity(self, op):
        """ Returns the sparsity of the given operator relative to the total Hilbert
        space size. """
        return self.state.sparsity(op)
    
    def configureOperator(self, node, trunc, basis):
        if node not in self.getNodeList():
            raise Exception("Node '%i' is not a valid circuit node." % node)
        if basis == "charge":
            self.state.circuit_operators.setNodeOperators(node, ChargeBasisOperators(node, trunc))
        elif basis == "oscillator":
            self.state.circuit_operators.setNodeOperators(
                node, OscillatorBasisOperators(node, trunc)
            )
        else:
            raise Exception("Unrecognized basis representation '%s'." % repr(basis))

    def getClassicalPotentialFunction(self):
        """Returns a function that takes the flux circuit degrees of freedom and the flux bias terms as scalars and
        returns the potential energy at those coordinates. It can be used with numpy arrays too."""
        builder = ClassicalPotentialBuilder(self)
        return builder.getPotentialFunction(), builder.getDefaultInputs()

    def substitute(self):
        """ Substitutes all current parameter values into the symbolic expressions, collecting
        the numerical arrays into a :class:`~pyscqed.simulation_state._NumericalParts`
        instance, and generates the expanded node operators. """
        self.state.substitute(self.SS.getSymbolValuesDict())
    
    def getLinearPart(self):
        parts = self.state.numerical_parts
        Q = self.state.circuit_operators.charge_op_vector + parts.charge_bias_vector
        P = self.state.circuit_operators.flux_op_vector + parts.inductive_flux_bias_vector

        # Get charging energy
        Hq = self.units.getPrefactor("Ec")*0.5*\
        util.mdot(Q.T, parts.inverse_capacitance_matrix, Q)[0, 0]

        # Get flux energy
        Hf = self.units.getPrefactor("El")*0.5*\
        util.mdot(P.T, parts.inverse_inductance_matrix, P)[0, 0]

        return Hq + Hf
    
    def getStaticJosephsonPart(self):
        Jvec = self.state.numerical_parts.josephson_vector

        # Need the branch DoFs in the possibly transformed representation
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector

        # Get the Josephson energy
        Hj_l = []
        Hj_r = []
        for i, edge in enumerate(self.SS.edges):
            if Jvec[i] == 0.0:
                continue
            prod1 = 0.0
            prod2 = 0.0
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                prod1 = 1.0
                prod2 = 1.0
                # Left
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] > 0:
                        prod1 *= self.getOperator(node, "disp")
                    else:
                        prod1 *= self.getOperator(node, "disp_adj")
                
                # Right
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] < 0:
                        prod2 *= self.getOperator(node, "disp")
                    else:
                        prod2 *= self.getOperator(node, "disp_adj")
            else:
                prod1 = 1.0
                prod2 = 1.0
                node = self.SS.node_map_rev[Pp[i].args[1]]
                if Pp[i].args[0] > 0:
                    prod1 *= self.getOperator(node, "disp")
                    prod2 *= self.getOperator(node, "disp_adj")
                else:
                    prod1 *= self.getOperator(node, "disp_adj")
                    prod2 *= self.getOperator(node, "disp")
        
            Hj_l.append(-0.5*self.units.getPrefactor("Ej")*Jvec[i]*prod1)
            Hj_r.append(-0.5*self.units.getPrefactor("Ej")*Jvec[i]*prod2)
        return Hj_l, Hj_r
    
    ###################################################################################################################
    #       Evaluables
    ###################################################################################################################
    
    def getHamiltonian(self) -> qt.Qobj:
        parts = self.state.numerical_parts
        Q = self.state.circuit_operators.charge_op_vector + parts.charge_bias_vector
        P = self.state.circuit_operators.flux_op_vector + parts.inductive_flux_bias_vector

        # Get charging energy
        Hq = self.units.getPrefactor("Ec")*0.5*\
        util.mdot(Q.T, parts.inverse_capacitance_matrix, Q)[0, 0]

        # Get flux energy
        Hf = self.units.getPrefactor("El")*0.5*\
        util.mdot(P.T, parts.inverse_inductance_matrix, P)[0, 0]

        # Need the branch DoFs in the possibly transformed representation
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector

        # Get the Josephson energy
        Hj = 0
        for i, edge in enumerate(self.SS.edges):
            if parts.josephson_vector[i] == 0.0:
                continue

            prod1 = parts.positive_flux_bias_exponentials[i]
            prod2 = parts.negative_flux_bias_exponentials[i]
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                # Left
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] > 0:
                        prod1 *= self.getOperator(node, "disp")
                    else:
                        prod1 *= self.getOperator(node, "disp_adj")
                
                # Right
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] < 0:
                        prod2 *= self.getOperator(node, "disp")
                    else:
                        prod2 *= self.getOperator(node, "disp_adj")
            else:
                node = self.SS.node_map_rev[Pp[i].args[1]]
                if Pp[i].args[0] > 0:
                    prod1 *= self.getOperator(node, "disp")
                    prod2 *= self.getOperator(node, "disp_adj")
                else:
                    prod1 *= self.getOperator(node, "disp_adj")
                    prod2 *= self.getOperator(node, "disp")
        
            Hj += -0.5*parts.josephson_vector[i]*(prod1 + prod2)
        Hj *= self.units.getPrefactor("Ej")

        # Total Hamiltonian
        return (Hq + Hf + Hj).tidyup(_qobj_atol)
    
    def getCurrentOperator(self, edge=None) -> qt.Qobj:
        parts = self.state.numerical_parts

        # Check edge
        if edge is None:
            raise Exception("No edge specified for branch current operator.")
        if edge not in self.getEdgeList():
            raise Exception("Edge %s not in the circuit. Check that it is in the right direction." % repr(edge))
        
        # Create the transformed branch vector and get the branch
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector
        i = self.getEdgeIndex(edge)
        
        # Inductive edge type should generally be favoured
        if self.getCircuitGraph().isInductiveEdge(edge):
            # Construct the branch flux operators from node representations
            if len(Pp[i].atoms()) > 2:
                sum1 = 0
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    sum1 += float(arg.args[0]) * self.getOperator(node, "flux")
            else:
                node = self.SS.node_map_rev[Pp[i].args[1]]
                sum1 = float(Pp[i].args[0]) * self.getOperator(node, "flux")
            
            # Take difference of node fluxes of corresponding branch and use *branch* inverse inductance matrix
            return self.units.getPrefactor("IopL") * sum1 * \
                parts.branch_inverse_inductance_matrix[i, i]
        
        elif self.getCircuitGraph().isJosephsonEdge(edge):
            
            # Get the Josephson operators
            prod1 = parts.positive_flux_bias_exponentials[i]
            prod2 = parts.negative_flux_bias_exponentials[i]
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                # Left
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] > 0:
                        prod1 *= self.getOperator(node, "disp")
                    else:
                        prod1 *= self.getOperator(node, "disp_adj")
                
                # Right
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] < 0:
                        prod2 *= self.getOperator(node, "disp")
                    else:
                        prod2 *= self.getOperator(node, "disp_adj")
            else:
                node = self.SS.node_map_rev[Pp[i].args[1]]
                if Pp[i].args[0] > 0:
                    prod1 *= self.getOperator(node, "disp")
                    prod2 *= self.getOperator(node, "disp_adj")
                else:
                    prod1 *= self.getOperator(node, "disp_adj")
                    prod2 *= self.getOperator(node, "disp")
        
            return 0.5j * self.units.getPrefactor("IopJ") * \
                parts.josephson_vector[i] * (prod1 - prod2)
        else:
            raise Exception("Edge %s is not current-carrying" % repr(edge))
    
    def getCurrentMatrixElement(self, energies: EigenvalueResult, vectors: EigenvectorResult, edge=None, elements=None) -> float:
        # Get the relevant operator
        Iop = self.getCurrentOperator(edge=edge)
        
        # Check the elements
        if elements is None:
            raise Exception("No elements specified for current matrix elements.")
        result = np.zeros(len(elements), dtype=np.float64)
        for i, indices in enumerate(elements):
            i1, i2 = indices
            result[i] = Iop.matrix_element(vectors.data[i1], vectors.data[i2]).real
        return result
    
    def getVoltageOperator(self, node=None):
        # Check node
        if node is None:
            raise Exception("No node specified for node voltage operator.")
        if node not in self.getNodeList():
            raise Exception("Node %s is not in the circuit." % repr(node))
        i = self.getNodeIndex(node)
        
        # Get charge superoperator including charge offsets
        parts = self.state.numerical_parts
        Q = self.state.circuit_operators.charge_op_vector + parts.charge_bias_vector

        # Use the inverse capacitance matrix
        return self.units.getPrefactor("Vop") * Q[i, 0] * \
            parts.inverse_capacitance_matrix[i, i]
    
    def getVoltageMatrixElement(self, energies: EigenvalueResult, vectors: EigenvectorResult, node=None, elements=None):
        # Get the relevant operator
        Vop = self.getVoltageOperator(node=node)
        
        # Check the elements
        if elements is None:
            raise Exception("No elements specified for voltage matrix elements.")
        result = np.zeros(len(elements), dtype=np.float64)
        for i, indices in enumerate(elements):
            i1, i2 = indices
            result[i] = Vop.matrix_element(vectors.data[i1], vectors.data[i2]).real
        return result
    
    def getChargingEnergies(self, node=None):
        Cinv = self.state.numerical_parts.inverse_capacitance_matrix
        if node is None:
            ret = {}
            for i, pos in enumerate(self.getNodeList()):
                ret[pos] = 0.5 * Cinv[i, i] * self.units.getPrefactor("Ec")
            return ret
        else:
            i = self.getNodeList().index(node)
            return 0.5 * Cinv[i, i] * self.units.getPrefactor("Ec")
    
    def getFluxEnergies(self, node=None):
        Linv = self.state.numerical_parts.inverse_inductance_matrix
        if node is None:
            ret = {}
            for i, pos in enumerate(self.getNodeList()):
                ret[pos] = 0.5 * Linv[i, i] * self.units.getPrefactor("El")
            return ret
        else:
            i = self.getNodeList().index(node)
            return 0.5 * Linv[i, i] * self.units.getPrefactor("El")
    
    def getJosephsonEnergies(self, edge=None):
        Jvec = self.state.numerical_parts.josephson_vector
        if edge is None:
            ret = {}
            for i, edge in enumerate(self.SS.edges):
                ret[edge] = Jvec[i] * self.units.getPrefactor("Ej")
            return ret
        else:
            i = self.SS.edges.index(edge)
            return Jvec[i] * self.units.getPrefactor("Ej")
    
    def getPhaseSlipEnergies(self, edge=None):
        Pvec = self.state.numerical_parts.phase_slip_vector
        if edge is None:
            ret = {}
            for i, edge in enumerate(self.SS.edges):
                ret[edge] = Pvec[i] * self.units.getPrefactor("Ep")
            return ret
        else:
            i = self.SS.edges.index(edge)
            return Pvec[i] * self.units.getPrefactor("Ep")
    
    def getResonatorResponse(
        self,
        energies: EigenvalueResult,
        vectors: EigenvectorResult,
        nmax=100,
        cpl_node=None
    ) -> np.array:
        # Save the derived parameter values for each sweep value
        if cpl_node is None:
            resonator_nodes = [node for node, value in self.SS.CG.resonators_cap.items() if value is not None]
            if len(resonator_nodes) == 1:
                cpl_node = resonator_nodes[0]
            else:
                raise RuntimeError("There are either no coupled resonators or too many to guess the node.")
        
        # Get the model parameters
        gC = self.SS.getParameterValue('g%ir' % cpl_node)
        wrl = self.SS.getParameterValue('f%irl' % cpl_node)
        
        # Get the operator associated with selected node
        index = self.getNodeList().index(cpl_node)
        Op = self.state.circuit_operators.charge_op_vector[index, 0] + \
            self.state.numerical_parts.charge_bias_vector[index, 0]
        
        # Get coupling terms
        E = energies.data - energies.data[0]
        V = vectors.data
        tmax = len(E)
        nmax = nmax + 1 + tmax
        norm = Op.matrix_element(V[0], V[1])
        g_list = []
        for i in range(tmax-1):
            g_list.append(Op.matrix_element(V[i], V[i+1]))
        
        # Apply the circuit derived prefactor
        g_list = gC * np.abs(np.array(g_list)/norm)
        
        # Diagonalise RWA strips
        order = np.zeros(tmax, dtype=int)
        diag_bare = np.array([-i*wrl + E[i] for i in range(tmax)])
        eigensolver_order = np.linalg.eigvalsh(np.diag(diag_bare))
        for i in range(tmax):
            index, = np.where(eigensolver_order == diag_bare[i])
            order[i] = index[0]
        Erwa = [0.0] * nmax
        diagonal_elements = None
        offdiagonal_elements = None
        strip_H = None
        e = None
        for n in range(nmax):
            diagonal_elements = np.array([(n-i)*wrl + E[i] for i in range(tmax)])
            offdiagonal_elements = np.array([g_list[i]*np.sqrt((n-i)*(n-i>0)) for i in range(tmax-1)])
            strip_H = (np.diag(diagonal_elements) + np.diag(offdiagonal_elements, 1) +
                       np.diag(offdiagonal_elements, -1))
            e = sc.linalg.eigvalsh(strip_H)
            Erwa[n] = np.array([e[i].real for i in order])
        
        return np.array(Erwa)
    
    ###################################################################################################################
    #       Diagonaliser Configuration
    ###################################################################################################################
    
    def setDiagConfig(
        self,
        eigvalues=5,
        get_vectors=False,
        sparse=False,
        sparsesolveropts={"sigma":None, "mode":"normal", "maxiter":None, "tol":1e-3, "which":"SA"}
    ):
        self.diagonalizer_config = {
            'kwargs':{
                'eigvalues':eigvalues, 
                'get_vectors':get_vectors, 
                'sparsesolveropts':sparsesolveropts
            }, 
            'sparse':sparse
        }
        
        # Choose the diagonalizer function and matrix conversion operation
        if sparse:
            self.diagonalizer_config['func'] = util.diagSparseH
        else:
            self.diagonalizer_config['func'] = util.diagDenseH
    
    def getDiagConfig(self):
        return self.diagonalizer_config
    
    def diagonalize(self, qobj: qt.Qobj) -> tuple[EigenvalueResult, EigenvectorResult | None]:
        result = self.diagonalizer_config['func'](qobj, **self.diagonalizer_config['kwargs'])
        if self.diagonalizer_config["kwargs"]["get_vectors"]:
            return (
                EigenvalueResult(data=result[0], source="diagonalize"),
                EigenvectorResult(data=result[1], source="diagonalize")
            )
        return EigenvalueResult(data=result, source="diagonalize"), None
    
    ###################################################################################################################
    #       Parameter Collection Wrapper Functions and Extended Functions
    ###################################################################################################################
    
    ## Set the value of a parameter.
    def setParameterValue(self, name, value):
        self.SS.setParameterValue(name, value)
        params = self.getParameterValuesDict()
        assert all(value is not None for value in params.values()), \
               "Some parameters do not have valid values. All parameters should be set first with the " \
               "setParameterValues function in one go."
        self.substitute()
    
    ## Get the value of a parameter.
    def getParameterValue(self, name):
        return self.SS.getParameterValue(name)
    
    ## Set many parameter values.
    def setParameterValues(self, *name_value_pairs):
        self.SS.setParameterValues(*name_value_pairs)
        params = self.getParameterValuesDict()
        assert all(value is not None for value in params.values()), "Not all parameters were set in this call. " \
               f"The missing parameters are {repr([name for name, value in params.items() if value is None])}"
        self.substitute()
    
    ## Get many parameter values.
    def getParameterValues(self, *names):
        return self.SS.getParameterValues(*names)
    
    ## Gets all parameters
    def getParameterValuesDict(self):
        return self.SS.getParameterValuesDict()
    
    def getParameterSweep(self, name):
        return self.SS.getParameterSweep(name)
    
    def getParameterNames(self):
        return self.SS.getParameterNamesList()
    
    def getPrefactor(self, name):
        return self.units.getPrefactor(name)
    
    ###################################################################################################################
    #       Parameter Sweep Functions
    ###################################################################################################################
    def newSweepConfig(self) -> SweepConfig:
        config = SweepConfig(self.SS)
        # Default evaluation graph
        config.setEvaluationGraph(HamiltonianSpectrum(self))
        return config
    
    def runSweep(self, sweep_config: SweepConfig, use_disk: bool = True) -> SweepResult:
        # FIXME: Determine if we should be saving the data to temp files rather than in RAM:
        # Use the diagonaliser configuration, the requested evaluation functions, and the total number of sweep setpoints that will be used.
        # Substitute the static parameters, leaving the swept symbols free
        self.state.substituteStatic(sweep_config.getStaticSymbols())

        evaluation_graph = sweep_config.getEvaluationGraph()
        if evaluation_graph is None:
            raise RuntimeError("An evaluation graph must be set.")

        results = []
        point_gen = sweep_config.getGenerator()
        with progress.bar.Bar('Solving', check_tty=False, max=sweep_config.getTotalCount()) as bar:
            for sweep_point in point_gen:
                # Set the parameter values, this updates the state for the next call to work
                self.SS.setParameterValues(sweep_point)
                self.state.substituteSwept(sweep_config.getSweptSymbols())

                # Compute evaluation graph
                default_inputs = evaluation_graph.getDefaultInputs()
                result = evaluation_graph.evaluate(default_inputs)

                if use_disk:
                    # Write to temp file
                    f = self.writePart(result)
                    results.append(f)
                else:
                    results.append(result)
                bar.next()
            bar.finish()

        if use_disk:
            return SweepResultFromDisk(sweep_config, results)
        else:
            return SweepResultFromMemory(sweep_config, results)
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    # FIXME: This causes issues when regenerating code
    def _set_parameter_units(self):
        
        # Get the unit prefactors
        Uf = self.units.getUnitPrefactor('Hz')
        Uo = self.units.getUnitPrefactor('Ohm')
        Uc = self.units.getUnitPrefactor('F')
        Ul = self.units.getUnitPrefactor('H')
        
        # Resonators
        if self.SS._has_resonators():
            for node, resonator in self.SS.CG.resonators_cap.items():
                if resonator is not None:
                    self.SS.addParameterisationPrefactor(resonator["Cr"], 1/(Uf*Uo*Uc))
                    self.SS.addParameterisationPrefactor(resonator["Lr"], Uo/(Uf*Ul))
                    self.SS.addParameterisationPrefactor(resonator["gC"], self.units.getPrefactor('ChgOscCpl'))
                    self.SS.addParameterisationPrefactor(resonator["frl"], self.units.getPrefactor('Freq'))
                    self.SS.addParameterisationPrefactor(resonator["Zrl"], self.units.getPrefactor('Impe'))


class ClassicalPotentialBuilder:
    """Used internally to build a numerical classical potential energy getter as a function of all the parameters.
    """
    def __init__(self, hamil):
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

    def _get_critical_currents(self):
        subs = self._hamil.SS.getSymbolValuesDict()
        assert all(val is not None for val in subs.values()), "All parameters need to be initialized"
        Jvec = self._hamil.SS.getJosephsonVector().transpose().subs(subs)
        return np.array(Jvec).astype(np.float64)

    def _get_symbolic_inductive_energy(self):
        P = self._hamil.SS.getFluxVector()
        Pe = self._hamil.SS.getFluxBiasVectorInd()
        expr = (P + Pe).transpose() * sy.Matrix(self._inverse_inductance_matrix) * (P + Pe)
        return expr[0, 0]

    def _get_input_format(self):
        # Get the DoF symbol-node map
        self._dof_map = {"phi%i" % n: sym[0] for n, sym in self._hamil.SS.node_dofs.items() if n > 0}

        # Get the bias symbols
        bias_terms = self._hamil.SS.getFluxBiasVector(mode="branch").free_symbols
        for sym in bias_terms:
            name = self._hamil.SS.getParameterFromSymbol(sym)
            self._dof_map[name] = sym

    def getDefaultInputs(self):
        return {name: 0.0 for name in self._dof_map}

    def getPotentialFunction(self):
        def potential(inputs):
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

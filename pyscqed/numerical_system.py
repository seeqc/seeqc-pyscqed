from typing import Any
import qutip as qt
import numpy as np
import sympy as sy
import scipy as sc
import networkx as nx
import progress.bar
import time

from .circuit_graph import CircuitGraph, CircuitGraphEdge
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
    def __init__(self, nsys: "NumericalSystem") -> None:
        super().__init__()
        self.addNode("Hamiltonian", fn=nsys.getHamiltonian, outputs=["qobj"])
        self.addNode("Spectrum", fn=nsys.diagonalize, outputs=["E"])
        self.addDependency("Hamiltonian", "Spectrum")


class SingleResonatorInteraction(EvaluationGraph):
    def __init__(self, nsys: "NumericalSystem") -> None:
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
    def __init__(self, symbolic_system: SymbolicSystem, unit: Units = Units("CQED1")) -> None:
        
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

    def getNodeList(self) -> list[int]:
        return self.SS.nodes

    def getNodeIndex(self, node: int) -> int:
        return self.SS.nodes.index(node)

    def getEdgeList(self) -> list[CircuitGraphEdge]:
        return self.SS.edges

    def getEdgeIndex(self, edge: CircuitGraphEdge) -> int:
        return self.SS.edges.index(edge)

    def getCircuitGraph(self) -> CircuitGraph:
        return self.SS.CG

    def getSymbolicSystem(self) -> SymbolicSystem:
        return self.SS

    def getOperator(self, node: int, kind: str) -> qt.Qobj:
        """ Returns the operator ``kind`` of the given node, expanded into the circuit
        Hilbert space.

        :param kind: one of ``"charge"``, ``"flux"``, ``"disp"`` or ``"disp_adj"``.
        """
        return self.state.getOperator(node, kind)

    def getHilbertSpaceSize(self) -> int:
        """ Returns the Hilbert space size considering all currently defined operator
        truncations. """
        return self.state.getHilbertSpaceSize()

    def sparsity(self, op: qt.Qobj) -> float:
        """ Returns the sparsity of the given operator relative to the total Hilbert
        space size. """
        return self.state.sparsity(op)

    def configureOperator(self, node: int, trunc: int, basis: str) -> None:
        if node not in self.getNodeList():
            raise Exception("Node '%i' is not a valid circuit node." % node)
        if basis == "charge":
            self.state.setNodeOperators(node, ChargeBasisOperators(node, trunc))
        elif basis == "oscillator":
            self.state.setNodeOperators(
                node, OscillatorBasisOperators(node, trunc)
            )
        else:
            raise Exception("Unrecognized basis representation '%s'." % repr(basis))

    # def getLinearPart(self) -> sy.Expr:
        # parts = self.state.numerical_parts
        # Q = self.state.circuit_operators.charge_op_vector + parts.charge_bias_vector
        # P = self.state.circuit_operators.flux_op_vector + parts.inductive_flux_bias_vector

        # # Get charging energy
        # Hq = self.units.getPrefactor("Ec")*0.5*\
        # util.mdot(Q.T, parts.inverse_capacitance_matrix, Q)[0, 0]

        # # Get flux energy
        # Hf = self.units.getPrefactor("El")*0.5*\
        # util.mdot(P.T, parts.inverse_inductance_matrix, P)[0, 0]

        # return Hq + Hf
    
    # def getStaticJosephsonPart(self) -> tuple[list[qt.Qobj], list[qt.Qobj]]:
        # Jvec = self.state.numerical_parts.josephson_vector

        # # Need the branch DoFs in the possibly transformed representation
        # Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector

        # # Get the Josephson energy
        # Hj_l = []
        # Hj_r = []
        # for i, edge in enumerate(self.SS.edges):
            # if Jvec[i] == 0.0:
                # continue
            # prod1 = 0.0
            # prod2 = 0.0
            # if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                # prod1 = 1.0
                # prod2 = 1.0
                # # Left
                # for arg in Pp[i].args:
                    # node = self.SS.node_map_rev[arg.args[1]]
                    # if arg.args[0] > 0:
                        # prod1 *= self.getOperator(node, "disp")
                    # else:
                        # prod1 *= self.getOperator(node, "disp_adj")
                
                # # Right
                # for arg in Pp[i].args:
                    # node = self.SS.node_map_rev[arg.args[1]]
                    # if arg.args[0] < 0:
                        # prod2 *= self.getOperator(node, "disp")
                    # else:
                        # prod2 *= self.getOperator(node, "disp_adj")
            # else:
                # prod1 = 1.0
                # prod2 = 1.0
                # node = self.SS.node_map_rev[Pp[i].args[1]]
                # if Pp[i].args[0] > 0:
                    # prod1 *= self.getOperator(node, "disp")
                    # prod2 *= self.getOperator(node, "disp_adj")
                # else:
                    # prod1 *= self.getOperator(node, "disp_adj")
                    # prod2 *= self.getOperator(node, "disp")
        
            # Hj_l.append(-0.5*self.units.getPrefactor("Ej")*Jvec[i]*prod1)
            # Hj_r.append(-0.5*self.units.getPrefactor("Ej")*Jvec[i]*prod2)
        # return Hj_l, Hj_r
    
    ###################################################################################################################
    #       Evaluables
    ###################################################################################################################
    
    def getHamiltonian(self) -> qt.Qobj:
        Q = self.state.getBiasedChargeOperatorVector()
        P = self.state.getBiasedFluxOperatorVector()

        # Get charging energy
        Hq = self.units.getPrefactor("Ec")*0.5*\
        util.mdot(Q.T, self.state.getInverseCapacitanceMatrix(), Q)[0, 0]

        # Get flux energy
        Hf = self.units.getPrefactor("El")*0.5*\
        util.mdot(P.T, self.state.getInverseInductanceMatrix(), P)[0, 0]

        # Need the branch DoFs in the possibly transformed representation
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector

        # Get the Josephson energy
        josephson_vector = self.state.getJosephsonVector()
        positive_exponentials = self.state.getPositiveFluxBiasExponentials()
        negative_exponentials = self.state.getNegativeFluxBiasExponentials()
        Hj = 0
        for i, edge in enumerate(self.SS.edges):
            if josephson_vector[i] == 0.0:
                continue

            prod1 = positive_exponentials[i]
            prod2 = negative_exponentials[i]
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
        
            Hj += -0.5*josephson_vector[i]*(prod1 + prod2)
        Hj *= self.units.getPrefactor("Ej")

        # Total Hamiltonian
        return (Hq + Hf + Hj).tidyup(_qobj_atol)
    
    def getCurrentOperator(self, edge: CircuitGraphEdge | None = None) -> qt.Qobj:
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
                self.state.getBranchInverseInductanceMatrix()[i, i]

        elif self.getCircuitGraph().isJosephsonEdge(edge):

            # Get the Josephson operators
            prod1 = self.state.getPositiveFluxBiasExponentials()[i]
            prod2 = self.state.getNegativeFluxBiasExponentials()[i]
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
                self.state.getJosephsonVector()[i] * (prod1 - prod2)
        else:
            raise Exception("Edge %s is not current-carrying" % repr(edge))
    
    def getCurrentMatrixElement(
        self,
        energies: EigenvalueResult,
        vectors: EigenvectorResult,
        edge: CircuitGraphEdge | None = None,
        elements: list[tuple[int, int]] | None = None
    ) -> np.ndarray:
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
    
    def getVoltageOperator(self, node: int | None = None) -> qt.Qobj:
        # Check node
        if node is None:
            raise Exception("No node specified for node voltage operator.")
        if node not in self.getNodeList():
            raise Exception("Node %s is not in the circuit." % repr(node))
        i = self.getNodeIndex(node)
        
        # Get charge superoperator including charge offsets
        Q = self.state.getBiasedChargeOperatorVector()

        # Use the inverse capacitance matrix
        return self.units.getPrefactor("Vop") * Q[i, 0] * \
            self.state.getInverseCapacitanceMatrix()[i, i]
    
    def getVoltageMatrixElement(
        self,
        energies: EigenvalueResult,
        vectors: EigenvectorResult,
        node: int | None = None,
        elements: list[tuple[int, int]] | None = None
    ) -> np.ndarray:
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
    
    def getChargingEnergies(self, node: int | None = None) -> dict[int, float] | float:
        Cinv = self.state.getInverseCapacitanceMatrix()
        if node is None:
            ret = {}
            for i, pos in enumerate(self.getNodeList()):
                ret[pos] = 0.5 * Cinv[i, i] * self.units.getPrefactor("Ec")
            return ret
        else:
            i = self.getNodeList().index(node)
            return 0.5 * Cinv[i, i] * self.units.getPrefactor("Ec")
    
    def getFluxEnergies(self, node: int | None = None) -> dict[int, float] | float:
        Linv = self.state.getInverseInductanceMatrix()
        if node is None:
            ret = {}
            for i, pos in enumerate(self.getNodeList()):
                ret[pos] = 0.5 * Linv[i, i] * self.units.getPrefactor("El")
            return ret
        else:
            i = self.getNodeList().index(node)
            return 0.5 * Linv[i, i] * self.units.getPrefactor("El")
    
    def getJosephsonEnergies(self, edge: CircuitGraphEdge | None = None) -> dict[CircuitGraphEdge, float] | float:
        Jvec = self.state.getJosephsonVector()
        if edge is None:
            ret = {}
            for i, edge in enumerate(self.SS.edges):
                ret[edge] = Jvec[i] * self.units.getPrefactor("Ej")
            return ret
        else:
            i = self.SS.edges.index(edge)
            return Jvec[i] * self.units.getPrefactor("Ej")

    def getResonatorResponse(
        self,
        energies: EigenvalueResult,
        vectors: EigenvectorResult,
        nmax: int = 100,
        cpl_node: int | None = None
    ) -> np.ndarray:
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
        Op = self.state.getBiasedChargeOperatorVector()[index, 0]
        
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
        eigvalues: int = 5,
        get_vectors: bool = False,
        sparse: bool = False,
        sparsesolveropts: dict[str, Any] = {"sigma":None, "mode":"normal", "maxiter":None, "tol":1e-3, "which":"SA"}
    ) -> None:
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
    
    def getDiagConfig(self) -> dict[str, Any]:
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
    def setParameterValue(self, name: str, value: float) -> None:
        self.SS.setParameterValue(name, value)
        params = self.getParameterValuesDict()
        assert all(value is not None for value in params.values()), \
               "Some parameters do not have valid values. All parameters should be set first with the " \
               "setParameterValues function in one go."
        self.state.substitute(self.SS.getSymbolValuesDict())
    
    ## Get the value of a parameter.
    def getParameterValue(self, name: str) -> float:
        return self.SS.getParameterValue(name)
    
    ## Set many parameter values.
    def setParameterValues(self, *name_value_pairs: str | float | dict[str, float]) -> None:
        self.SS.setParameterValues(*name_value_pairs)
        params = self.getParameterValuesDict()
        assert all(value is not None for value in params.values()), "Not all parameters were set in this call. " \
               f"The missing parameters are {repr([name for name, value in params.items() if value is None])}"
        self.state.substitute(self.SS.getSymbolValuesDict())
    
    ## Get many parameter values.
    def getParameterValues(self, *names: str) -> dict[str, float]:
        return self.SS.getParameterValues(*names)

    ## Gets all parameters
    def getParameterValuesDict(self) -> dict[str, float]:
        return self.SS.getParameterValuesDict()

    def getParameterSweep(self, name: str) -> np.ndarray:
        return self.SS.getParameterSweep(name)

    def getParameterNames(self) -> list[str]:
        return self.SS.getParameterNamesList()

    def getPrefactor(self, name: str) -> float:
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
    def _set_parameter_units(self) -> None:
        
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

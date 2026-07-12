import pytest

import qutip as qt
import numpy as np

from pyscqed.circuit_graph import CircuitGraph
from pyscqed.symbolic_system import SymbolicSystem
from pyscqed.numerical_system import NumericalSystem
from pyscqed.evaluation_graph import EvaluationGraph
from pyscqed.physical_constants import hbar, phi0, e
from pyscqed.util import mdot


def get_single_node_graph() -> CircuitGraph:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")
    return graph


def get_numerical_system(graph: CircuitGraph) -> NumericalSystem:
    # NOTE: The system simulated here is not configured for generating representative physical results,
    # they are only there to check numerical consistency
    symbolic = SymbolicSystem(graph)
    hamil = NumericalSystem(symbolic)
    hamil.configureOperator(1, 40, "charge")
    return hamil


def get_flux_qubit(operator_basis: str) -> NumericalSystem:
    Ca = 60.0  # fF/um^2
    Jc = 3.0  # uA/um^2
    Aj = 0.6  # um^2
    graph = get_single_node_graph()
    graph.addFluxBias("I", "Z")
    circuit = SymbolicSystem(graph)
    hamil = NumericalSystem(circuit)
    hamil.configureOperator(1, 100, operator_basis)
    hamil.setParameterValues(
        "L", 570.0,
        "I", Jc*Aj,
        "C", Ca*Aj,
        "phiZ", 0.5
    )
    return hamil


def get_resonator(operator_basis: str) -> NumericalSystem:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")
    circuit = SymbolicSystem(graph)
    hamil = NumericalSystem(circuit)
    hamil.configureOperator(1, 100, operator_basis)
    hamil.setParameterValues(
        "L", 500.0,
        "C", 150.0
    )
    return hamil


# TODO: Should this be a patch of `getHamiltonian`?
def get_linear_hamiltonian(hamil: NumericalSystem, basis_transform: bool) -> qt.Qobj:
    if basis_transform:
        Q = hamil.state.getBiasedChargeOperatorVector()
        P = hamil.state.getBiasedFluxOperatorVector()
    else:
        Q = (hamil.state.circuit_operators.charge_op_vector
             + hamil.state.numerical_parts.charge_bias_vector)
        P = (hamil.state.circuit_operators.flux_op_vector
             + hamil.state.numerical_parts.inductive_flux_bias_vector)

    # Get charging energy
    Hq = hamil.units.getPrefactor("Ec")*0.5*\
    mdot(Q.T, hamil.state.getInverseCapacitanceMatrix(), Q)[0, 0]

    # Get flux energy
    Hf = hamil.units.getPrefactor("El")*0.5*\
    mdot(P.T, hamil.state.getInverseInductanceMatrix(), P)[0, 0]
    
    return Hq + Hf


def test_sweep_one_dimension():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH
    )

    expected_spectrum_sweep = [
        [59.95010688, 45.30354992, 36.57241861],
        [219.58210001, 175.64339355, 149.45048272],
        [379.20831265, 305.97937504, 262.32564643],
        [538.82879099, 436.31151958, 375.19792611],
        [698.44358094, 566.63985224, 488.06733808]
    ]

    trace = np.linspace(20.0, 40.0, 3)
    sweep = hamil.newSweepConfig()
    sweep.add("C", trace)

    result_disk = hamil.runSweep(sweep)
    spectrum = result_disk.getNumericalOutput("C")
    assert np.allclose(spectrum.T, expected_spectrum_sweep, rtol=0, atol=1e-6)
    traces = result_disk.getInputPoints("C")
    assert np.array_equal(traces, trace)

    result_mem = hamil.runSweep(sweep, use_disk=False)
    spectrum = result_mem.getNumericalOutput("C")
    assert np.allclose(spectrum.T, expected_spectrum_sweep, rtol=0, atol=1e-6)
    traces = result_disk.getInputPoints("C")
    assert np.array_equal(traces, trace)


def test_sweep_two_dimensions():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH 
    )

    expected_spectrum_sweep1 = [
        [59.95010688, 45.30354992, 36.57241861],
        [219.58210001, 175.64339355, 149.45048272],
        [379.20831265, 305.97937504, 262.32564643],
        [538.82879099, 436.31151958, 375.19792611],
        [698.44358094, 566.63985224, 488.06733808]
    ]
    expected_spectrum_sweep2 = [
        [53.03927077, 39.66091887, 31.68577904],
        [198.84901805, 158.71511714, 134.79027614],
        [344.65184476, 277.7646906, 237.89129967],
        [490.44781127, 396.80967219, 340.98887104],
        [636.23697759, 515.85009468, 444.0830116 ]
    ]

    sweep = hamil.newSweepConfig()
    trace1 = np.linspace(20.0, 40.0, 3)
    trace2 = np.linspace(50.0, 60.0, 2)
    sweep.add("C", trace1)
    sweep.add("L", trace2)
    result = hamil.runSweep(sweep)

    with pytest.raises(
        ValueError,
        match="Insufficient independent and static variables to retrieve sweep result data."
    ):
        result.getOutputs("C")

    with pytest.raises(
        ValueError,
        match="Independent variable \"I\" not present in sweep."
    ):
        result.getOutputs("I", {"L": 50.0})

    spectrum1 = result.getNumericalOutput("C", {"L": 50.0})
    spectrum2 = result.getNumericalOutput("C", {"L": 60.0})
    assert np.allclose(spectrum1.T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(spectrum2.T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces1 = result.getInputPoints("C")
    assert np.array_equal(traces1, trace1)

    # Test multi-dimensional sweep retrieval
    CL = result.getNumericalOutput(["C", "L"])
    assert np.allclose(CL[:, 0].T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(CL[:, 1].T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces = result.getInputMesh(["C", "L"])
    assert np.array_equal(traces["C"][:, 0], trace1)
    assert np.array_equal(traces["L"][0, :], trace2)

    # Test that the independent variable input order correctly formats the output
    CL = result.getNumericalOutput(["L", "C"])
    assert np.allclose(CL[0].T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(CL[1].T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces = result.getInputMesh(["L", "C"])
    assert np.array_equal(traces["C"][0, :], trace1)
    assert np.array_equal(traces["L"][:, 0], trace2)

    result_mem = hamil.runSweep(sweep, use_disk=False)
    CL = result_mem.getNumericalOutput(["L", "C"])
    assert np.allclose(CL[0].T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(CL[1].T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces = result.getInputMesh(["L", "C"])
    assert np.array_equal(traces["C"][0, :], trace1)
    assert np.array_equal(traces["L"][:, 0], trace2)


def test_sweep_arb_eval_graph():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH
    )

    input_array = np.arange(6).reshape((2, 3))

    def generator() -> np.ndarray:
        return input_array

    def manipulator(array: np.ndarray) -> np.ndarray:
        return array.T

    def final(array: np.ndarray) -> int:
        return array.size

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Generator", fn=generator, outputs=["array"])
    eval_graph.addNode("Manipulator", fn=manipulator, outputs=["array"])
    eval_graph.addNode("Final", fn=final, outputs=["size"])
    eval_graph.addDependency("Generator", "Manipulator", preserve_source_outputs=True)
    eval_graph.addDependency("Manipulator", "Final", preserve_source_outputs=True)

    trace = np.linspace(20.0, 40.0, 2)
    sweep = hamil.newSweepConfig()
    sweep.setEvaluationGraph(eval_graph)
    sweep.add("C", trace)
    result = hamil.runSweep(sweep)

    # All evaluation results by default
    all_results = result.getOutputs("C")

    # Individual evaluation results
    size_result = result.getNumericalOutput("C", eval_output=("Final", "size"))
    assert np.array_equal(size_result, np.array([6., 6.]))
    assert np.array_equal(size_result, all_results[("Final", "size")])

    gen_result = result.getNumericalOutput("C", eval_output=("Generator", "array"))
    assert np.array_equal(gen_result, np.array([input_array, input_array]))
    assert np.array_equal(gen_result, all_results[("Generator", "array")])

    man_result = result.getNumericalOutput("C", eval_output=("Manipulator", "array"))
    assert np.array_equal(man_result, np.array([input_array.T, input_array.T]))
    assert np.array_equal(man_result, all_results[("Manipulator", "array")])


def test_sweep_axis_reordering():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH
    )

    def generator() -> list[float]:
        return [hamil.getParameterValue("C"), hamil.getParameterValue("L")]

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Generator", fn=generator, outputs=["array"])

    # Sweep one dimension, no reordering
    sweep = hamil.newSweepConfig()
    sweep.setEvaluationGraph(eval_graph)
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    result = hamil.runSweep(sweep)
    all_results = result.getOutputs("C")
    assert all_results[('Generator', 'array')][0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1] == [40.0, 50.0]

    # Sweep two dimensions, but request a trace that reorders the L axis from 1 to 0
    sweep = hamil.newSweepConfig()
    sweep.setEvaluationGraph(eval_graph)
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    sweep.add("L", np.linspace(50.0, 60.0, 2))
    result = hamil.runSweep(sweep)
    all_results = result.getOutputs("L", {"C": 20})
    assert all_results[('Generator', 'array')][0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1] == [20.0, 60.0]
    all_results = result.getOutputs("L", {"C": 40})
    assert all_results[('Generator', 'array')][0] == [40.0, 50.0]
    assert all_results[('Generator', 'array')][1] == [40.0, 60.0]

    # Original axes
    all_results = result.getOutputs(["C", "L"])
    assert all_results[('Generator', 'array')][0, 0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1, 0] == [40.0, 50.0]

    # Swapped axes
    all_results = result.getOutputs(["L", "C"])
    assert all_results[('Generator', 'array')][0, 0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1, 0] == [20.0, 60.0]


def create_test_numerical_system() -> NumericalSystem:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")

    hamil = NumericalSystem(SymbolicSystem(graph))
    hamil.configureOperator(1, 40, "charge")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    return hamil


def test_get_numerical_output_non_numpy_array():
    hamil = create_test_numerical_system()

    def make_list() -> list:
        return [object(), object()]

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Producer", fn=make_list, outputs=["items"])

    sweep_config = hamil.newSweepConfig()
    sweep_config.setEvaluationGraph(eval_graph)
    sweep_config.add("C", np.linspace(20.0, 40.0, 3))
    sweep = hamil.runSweep(sweep_config)

    # This must not raise ValueError
    items = sweep.getNumericalOutput("C", eval_output=("Producer", "items"))
    assert items.shape == (3,)
    assert isinstance(items[0], list)
    assert len(items[0]) == 2


def test_get_numerical_output_eigenvalues_with_vectors():
    hamil = create_test_numerical_system()
    hamil.setDiagConfig(get_vectors=True, eigvalues=5)

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Hamiltonian", fn=hamil.getHamiltonian, outputs=["qobj"])
    eval_graph.addNode("Spectrum", fn=hamil.diagonalize, outputs=["energies", "vectors"])
    eval_graph.addDependency("Hamiltonian", "Spectrum")

    sweep_config = hamil.newSweepConfig()
    sweep_config.setEvaluationGraph(eval_graph)
    sweep_config.add("C", np.linspace(20.0, 40.0, 3))
    sweep = hamil.runSweep(sweep_config)

    energies = sweep.getNumericalOutput("C", eval_output=("Spectrum", "energies"))
    assert isinstance(energies, np.ndarray)
    assert energies.shape == (3, 5)

    vectors = sweep.getNumericalOutput("C", eval_output=("Spectrum", "vectors"))
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 5)


def test_spectrum_flux_dependence_independent_of_operator_basis():
    hamil = get_flux_qubit("charge")
    sweep_config = hamil.newSweepConfig()
    sweep_config.add('phiZ', np.linspace(-1.0, 1.0, 11))
    sweep = hamil.runSweep(sweep_config)
    E1 = sweep.getNumericalOutput('phiZ')
    
    hamil = get_flux_qubit("oscillator")
    sweep_config = hamil.newSweepConfig()
    sweep_config.add('phiZ', np.linspace(-1.0, 1.0, 11))
    sweep = hamil.runSweep(sweep_config)
    E2 = sweep.getNumericalOutput('phiZ')

    assert np.allclose(E1, E2, atol=1e-5, rtol=0)


def test_spectrum_basis_independence():
    hamil = get_flux_qubit("charge")
    sweep_config = hamil.newSweepConfig()
    sweep_config.add('phiZ', np.linspace(-1.0, 1.0, 11))
    sweep_config.add('L', np.linspace(500, 600, 3))
    sweep = hamil.runSweep(sweep_config)
    E1 = sweep.getNumericalOutput(['phiZ', 'L'])
    
    hamil = get_flux_qubit("oscillator")
    sweep_config = hamil.newSweepConfig()
    sweep_config.add('phiZ', np.linspace(-1.0, 1.0, 11))
    sweep_config.add('L', np.linspace(500, 600, 3))
    sweep = hamil.runSweep(sweep_config)
    E2 = sweep.getNumericalOutput(['phiZ', 'L'])

    assert np.allclose(E1, E2, atol=1e-5, rtol=0)


def test_spectrum_circuit_element_dependence_independent_of_operator_basis():
    hamil = get_resonator("charge")
    sweep_config = hamil.newSweepConfig()
    sweep_config.add('L', np.linspace(400, 600, 3))
    sweep = hamil.runSweep(sweep_config)
    E1 = sweep.getNumericalOutput('L')

    hamil = get_resonator("oscillator")
    sweep_config = hamil.newSweepConfig()
    sweep_config.add('L', np.linspace(400, 600, 3))
    sweep = hamil.runSweep(sweep_config)
    E2 = sweep.getNumericalOutput('L')

    assert np.allclose(E1.T - E1.T[0], E2.T - E2.T[0], atol=1e-5, rtol=0)


def test_oscillator_parameters_are_consistent_independent_of_operator_basis():
    charge_hamil = get_resonator("charge")

    # Verify the oscillator parameters match
    H = charge_hamil.units.getUnitPrefactor("H")
    F = charge_hamil.units.getUnitPrefactor("F")
    L = charge_hamil.getParameterValue("L") * H
    C = charge_hamil.getParameterValue("C") * F

    Hosc = charge_hamil.getHamiltonian()
    result, _ = charge_hamil.diagonalize(Hosc)
    f_diag = (result.data[1] - result.data[0]) * 1e9

    f = 1 / np.sqrt(L * C) / 2 / np.pi
    assert np.isclose(f_diag, f, atol=2e1, rtol=0)

    osc_hamil = get_resonator("oscillator")

    H = osc_hamil.units.getUnitPrefactor("H")
    F = osc_hamil.units.getUnitPrefactor("F")
    L = osc_hamil.getParameterValue("L") * H
    C = osc_hamil.getParameterValue("C") * F

    Hosc = osc_hamil.getHamiltonian()
    result, _ = osc_hamil.diagonalize(Hosc)
    f_diag = (result.data[1] - result.data[0]) * 1e9

    f = 1 / np.sqrt(L * C) / 2 / np.pi
    assert np.isclose(f_diag, f, atol=2e1, rtol=0)
    Z = np.sqrt(L / C)

    Zpref = np.sqrt(H / F)
    fpref = np.sqrt(1 / H / F)
    Zder = osc_hamil.getParameterValue("Zosc1") * Zpref
    fder = osc_hamil.getParameterValue("fosc1") * fpref / 2 / np.pi

    assert np.isclose(Zder, Z, atol=1e-5, rtol=0)
    assert np.isclose(fder, f, atol=2e1, rtol=0)


@pytest.mark.parametrize("basis", ["oscillator", "charge"])
def test_spectrum_bias_independence(basis):
    # Linear resonator with two parallel inductors forming a loop
    graph = CircuitGraph()
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L1")
    graph.addBranch(0, 1, "L2")

    # Statically biasing the loop should only create an energy offset and not
    # affect the energy spectrum
    graph.addFluxBias("L1", "Z")
    graph.addChargeBias(1, "X")
    circuit = SymbolicSystem(graph)

    # Calculate expected resonant frequency
    L = 500.0e-12  # H
    C = 1200.0e-15  # F
    expected_frequency = 1 / np.sqrt(L * C) / 2 / np.pi * 1e-9  # GHz
    hamil = NumericalSystem(circuit)
    hamil.configureOperator(1, 60, basis)
    hamil.setParameterValues(
        "L1", 2 * L * 1e12,
        "L2", 2 * L * 1e12,
        "C", C * 1e15,
        "phiZ", 0.0,
        "QX", 0.0
    )

    # Calculate expected flux energy offset
    phase_offset = 1.0  # phi0
    energy_offset = (phase_offset * phi0)**2 / (2 * L * hbar * 2 * np.pi) * 1e-9

    E1, V1 = hamil.getHamiltonian().eigenstates()
    assert np.isclose(E1[1] - E1[0], expected_frequency, atol=1e-5, rtol=0)
    assert np.isclose(E1[2] - E1[1], expected_frequency, atol=1e-5, rtol=0)

    # Appying a bias creates an energy offset LI^2/2 == phi^2/2L
    hamil.setParameterValue("phiZ", phase_offset)

    H = hamil.getHamiltonian()
    E2 = H.eigenenergies()
    assert np.isclose(E2[1] - E2[0], expected_frequency, atol=1e-5, rtol=0)
    assert np.isclose(E2[2] - E2[1], expected_frequency, atol=1e-5, rtol=0)

    # To get the energy offset term, we must construct the Hamiltonian without the basis shifting
    H = get_linear_hamiltonian(hamil, basis_transform=False)
    assert np.isclose(qt.expect(H, V1[0]) - E1[0], energy_offset, atol=1e-5, rtol=0)

    # Calculate expected charge energy offset
    charge_offset = 1.0  # 2e
    energy_offset = (charge_offset * 2 * e)**2 / (2 * C * hbar * 2 * np.pi) * 1e-9

    # The same is true for charge offsets
    hamil.setParameterValues(
        "phiZ", 0.0,
        "QX", charge_offset
    )

    H = hamil.getHamiltonian()
    E3 = H.eigenenergies()
    assert np.isclose(E3[1] - E3[0], expected_frequency, atol=1e-5, rtol=0)
    assert np.isclose(E3[2] - E3[1], expected_frequency, atol=1e-5, rtol=0)

    H = get_linear_hamiltonian(hamil, basis_transform=False)
    assert np.isclose(qt.expect(H, V1[0]) - E1[0], energy_offset, atol=1e-5, rtol=0)

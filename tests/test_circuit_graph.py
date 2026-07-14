import itertools
import os

import pytest

import numpy as np

from pyscqed.circuit_graph import CircuitGraph


def do_all_branch_type_tests_on_branch(G, edge, expected_type):
    fn = [
        G.isCapacitiveEdge,
        G.isInductiveEdge,
        G.isJosephsonEdge
    ]
    for i in range(3):
        if i == G._element_prefixes.index(expected_type):
            assert fn[i](edge)
        else:
            assert not fn[i](edge)


def get_simple_test_graph():
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I1")
    graph.addBranch(0, 1, "I2")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")
    return graph


def all_loops_are_unique(G):
    """There should be at least one distinct branch per loop."""
    available = {}
    for i, loop in G.sc_loops.items():
        curr = set(loop)
        for j, other_loop in G.sc_loops.items():
            if i == j:
                continue
            curr -= set(other_loop)
        assert len(curr) > 0


def plot_all_graphs(G, prefix):
    for t in G._subgraphs:
        G.drawGraphViz(graph=t, filename=("%s_%s" % (prefix, t)))


def test_graph_branch_rules():
    # Test that branch types can be correctly identified
    for i in range(3):
        graph = CircuitGraph()
        graph.addBranch(0, 1, graph._element_prefixes[i])
        do_all_branch_type_tests_on_branch(graph, (0, 1, 0), graph._element_prefixes[i])

    # Cannot add invalid components
    with pytest.raises(ValueError):
        graph.addBranch(0, 1, "J")

    # Test that the correct number of superconducting loops are detected
    for perm in itertools.permutations(["I", "L"], 2):
        graph = CircuitGraph()
        for item in perm:
            graph.addBranch(0, 1, item)
        assert len(graph.closure_branches) == 1
        assert len(graph.sc_loops) == 1
        all_loops_are_unique(graph)

    for perm in itertools.permutations(["I1", "I2", "L"], 3):
        graph = CircuitGraph()
        for item in perm:
            graph.addBranch(0, 1, item)
        assert len(graph.closure_branches) == 2
        assert len(graph.sc_loops) == 2
        all_loops_are_unique(graph)

    for perm in itertools.permutations(["I1", "I2", "I3", "L"], 4):
        graph = CircuitGraph()
        for item in perm:
            graph.addBranch(0, 1, item)
        assert len(graph.closure_branches) == 3
        assert len(graph.sc_loops) == 3
        all_loops_are_unique(graph)

    # Floating single loop
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I1")
    graph.addBranch(1, 2, "L1")
    graph.addBranch(1, 2, "L2")
    assert len(graph.closure_branches) == 1
    assert len(graph.sc_loops) == 1
    all_loops_are_unique(graph)

    # No conducting loops
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(1, 2, "C")
    graph.addBranch(1, 2, "L")
    assert len(graph.closure_branches) == 0
    assert len(graph.sc_loops) == 0
    all_loops_are_unique(graph)

    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(1, 2, "C")
    graph.addBranch(0, 2, "L")
    assert len(graph.closure_branches) == 0
    assert len(graph.sc_loops) == 0
    all_loops_are_unique(graph)

    # Large loop
    graph = CircuitGraph()
    N = 6
    for i in range(N):
        graph.addBranch(
            i, (i + 1) % N,
            "%s%i" % (graph._element_prefixes[np.random.randint(2) + 1], i)
        )
    assert len(graph.closure_branches) == 1
    assert len(graph.sc_loops) == 1
    all_loops_are_unique(graph)

    # Loops in series forming a larger loop
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I1")
    graph.addBranch(0, 1, "C1")
    graph.addBranch(0, 1, "L1") # 1 loop on (0, 1, k)
    graph.addBranch(1, 2, "L2")
    graph.addBranch(1, 2, "I2")
    graph.addBranch(1, 2, "C2") # 1 loop on (1, 2, k)
    graph.addBranch(0, 2, "I3")
    graph.addBranch(0, 2, "I4")
    graph.addBranch(0, 2, "L3") # 2 loops on (0, 2, k)
    assert len(graph.closure_branches) == 5
    assert len(graph.sc_loops) == 5
    all_loops_are_unique(graph)

    # Larger loops sharing a branch (3 in this case)
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I1")
    graph.addBranch(1, 2, "L1a")
    graph.addBranch(2, 0, "L1b")
    graph.addBranch(1, 3, "L2a")
    graph.addBranch(3, 0, "L2b")
    graph.addBranch(1, 4, "L3a")
    graph.addBranch(4, 0, "L3b")
    assert len(graph.closure_branches) == 3
    assert len(graph.sc_loops) == 3
    all_loops_are_unique(graph)

    # Larger loops sharing a loop (3 in this case)
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I1")
    graph.addBranch(0, 1, "I2")
    graph.addBranch(1, 2, "L1a")
    graph.addBranch(2, 0, "L1b")
    graph.addBranch(1, 3, "L2a")
    graph.addBranch(3, 0, "L2b")
    graph.addBranch(1, 4, "L3a")
    graph.addBranch(4, 0, "L3b")
    assert len(graph.closure_branches) == 4
    assert len(graph.sc_loops) == 4
    all_loops_are_unique(graph)

    # Larger loops sharing a large loop (2 in this case)
    graph = CircuitGraph()
    graph.addBranch(0, 1, "Lma")
    graph.addBranch(1, 2, "Ima")
    graph.addBranch(0, 3, "Lmb")
    graph.addBranch(3, 2, "Imb")
    graph.addBranch(0, 4, "I1a")
    graph.addBranch(4, 2, "L1a")
    graph.addBranch(0, 5, "I2a")
    graph.addBranch(5, 2, "L2a")
    assert len(graph.closure_branches) == 3
    assert len(graph.sc_loops) == 3
    all_loops_are_unique(graph)


def test_component_listers():
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I1")
    graph.addBranch(0, 1, "C1")
    graph.addBranch(0, 1, "I2")
    graph.addBranch(0, 1, "C2")
    graph.addBranch(1, 2, "L1a")
    graph.addBranch(2, 0, "L1b")
    graph.addBranch(1, 3, "L2a")
    graph.addBranch(3, 0, "L2b")
    graph.addBranch(1, 4, "L3a")
    graph.addBranch(4, 0, "L3b")

    inds = {"L1a", "L1b", "L2a", "L2b", "L3a", "L3b"}
    jjs = {"I1", "I2"}
    caps = {"C1", "C2"}

    assert set(graph.getInductiveEdges()) == inds
    assert set(graph.getJosephsonEdges()) == jjs
    assert set(graph.getCapacitiveEdges()) == caps


def test_flux_biasing_rules():
    graph = get_simple_test_graph()

    # Can add a flux bias to a JJ branch without loading
    graph.addFluxBias("I1", "Z1")

    # Cannot add a flux bias again to the same JJ branch
    with pytest.raises(ValueError):
        graph.addFluxBias("I1", "Z1")

    # Cannot use the same suffix twice
    with pytest.raises(ValueError):
        graph.addFluxBias("I2", "Z1")

    # Can add a flux bias to the second JJ branch without loading
    graph.addFluxBias("I2", "Z2")

    graph = get_simple_test_graph()

    # Cannot add a flux bias to a JJ branch with loading
    with pytest.raises(TypeError):
        graph.addFluxBias("I1", "Z1", "M1")

    # Can add a flux bias to a inductive branch with loading
    graph.addFluxBias("L", "Z", "M")


def test_charge_biasing_rules():
    graph = get_simple_test_graph()

    # Can add a charge bias to a node without loading
    graph.addChargeBias(1, "XY")

    # Cannot add a charge bias again to the same node
    with pytest.raises(ValueError):
        graph.addChargeBias(1, "XY")

    # Cannot use the same suffix twice
    # Note node 0 should be protected, as it currently always represents ground, however
    # it isn't clear how to enforce that without inconvenience to the user at the moment.
    with pytest.raises(ValueError):
        graph.addChargeBias(0, "XY")


def test_flux_biasing_on_loops():
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I1")
    graph.addBranch(1, 2, "L1")
    graph.addBranch(1, 2, "L2")

    # Cannot add a flux bias to an element that is not part of a superconducting loop
    with pytest.raises(TypeError):
        graph.addFluxBias("I1", "Z1")


def test_mutual_inductance():
    graph = CircuitGraph()
    graph.addBranch(0, 1, "L1")
    graph.addBranch(0, 1, "I1")
    graph.addBranch(0, 1, "C1")
    graph.addBranch(0, 2, "L2")
    graph.addBranch(0, 2, "I2")
    graph.addBranch(0, 2, "C2")

    # Can couple inductive branches
    graph.coupleBranchesInductively("L1", "L2", "M")

    # Cannot couple non inductive branches
    with pytest.raises(TypeError):
        graph.coupleBranchesInductively("I1", "L2", "M1")
    with pytest.raises(TypeError):
        graph.coupleBranchesInductively("L1", "I2", "M2")
    with pytest.raises(TypeError):
        graph.coupleBranchesInductively("I1", "I2", "M3")

    # Cannot use the same mutual name twice
    with pytest.raises(ValueError):
        graph.coupleBranchesInductively("L1", "L2", "M")

    # Cannot couple the same inductors twice
    with pytest.raises(ValueError):
        graph.coupleBranchesInductively("L1", "L2", "M4")


def test_loop_biases_irreducible_paths():
    graph = CircuitGraph()
    graph.addBranch(0, 1, "CL")
    graph.addBranch(0, 2, "CI")
    graph.addBranch(0, 7, "CR")
    graph.addBranch(1, 7, "Ll")
    graph.addBranch(1, 4, "LBL")
    graph.addBranch(1, 3, "LTL")
    graph.addBranch(7, 6, "LBR")
    graph.addBranch(7, 5, "LTR")
    graph.addBranch(2, 4, "IBL")
    graph.addBranch(2, 3, "ITL")
    graph.addBranch(2, 6, "IBR")
    graph.addBranch(2, 5, "ITR")
    graph.addFluxBias("ITL", "L")
    graph.addFluxBias("ITR", "R")
    graph.addFluxBias("Ll", "Z")

    print(graph.loop_biases)


def test_loop_biases_parallel_junctions():
    graph = CircuitGraph()
    graph.addBranch(0, 1, "C1")
    graph.addBranch(0, 1, "I1")
    graph.addBranch(1, 2, "C2a")
    graph.addBranch(1, 2, "I2a")
    graph.addBranch(1, 2, "C2b")
    graph.addBranch(1, 2, "I2b")
    graph.addBranch(1, 2, "Csh")
    graph.addBranch(0, 2, "C3")
    graph.addBranch(0, 2, "I3")
    graph.addFluxBias("I3", "Z")
    graph.addFluxBias("I2a", "X")

    print(graph.loop_biases)

import pytest

import numpy as np

from pyscqed.evaluation_graph import EvaluationGraph


def node_test_fn1(input: int):
    return input

def node_test_fn2(arg: int) -> bool:
    return arg == 1

def node_test_fn3(arg: int) -> float:
    return 0.0 if arg == 1 else 10.0

def node_test_fn4(switch: bool) -> tuple[float, float]:
    if switch:
        return 0.0, 1.0
    return 1.0, 0.0


def test_evaluation_graph_series_nodes():
    graph = EvaluationGraph()
    graph.add_node("A", fn=node_test_fn1, outputs=["arg"])
    graph.add_node("B", fn=node_test_fn2, outputs=["switch"])
    graph.add_node("C", fn=node_test_fn4, outputs=["final1", "final2"])
    graph.add_dependency("A", "B", preserve_source_outputs=True)
    graph.add_dependency("B", "C", preserve_source_outputs=True)
    inputs = {
        "A": {
            "input": 1
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result["A"]["arg"] == 1
    assert result["B"]["switch"] == True
    assert result["C"]["final1"] == 0.0
    assert result["C"]["final2"] == 1.0

    inputs = {
        "A": {
            "input": 0
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result["A"]["arg"] == 0
    assert result["B"]["switch"] == False
    assert result["C"]["final1"] == 1.0
    assert result["C"]["final2"] == 0.0


def test_evaluation_graph_fanout():
    graph = EvaluationGraph()
    graph.add_node("A", fn=node_test_fn1, outputs=["arg"])
    graph.add_node("B", fn=node_test_fn2, outputs=["final1"])
    graph.add_node("C", fn=node_test_fn3, outputs=["final2"])
    graph.add_dependency("A", "B", preserve_source_outputs=True)
    graph.add_dependency("A", "C", preserve_source_outputs=True)
    inputs = {
        "A": {
            "input": 1
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result["A"]["arg"] == 1
    assert result["B"]["final1"] == True
    assert result["C"]["final2"] == 0.0


def test_evaluation_dag():
    graph = EvaluationGraph()
    graph.add_node("A", fn=node_test_fn1, outputs=["arg"])
    graph.add_node("B", fn=node_test_fn2, outputs=["final1"])
    graph.add_dependency("A", "B")
    graph.add_dependency("B", "A")
    inputs = {
        "A": {
            "input": 1
        }
    }
    with pytest.raises(TypeError, match="The evaluation graph structure is not a DAG."):
        graph.evaluate(inputs=inputs)


def test_evaluation_output_preservation():
    graph = EvaluationGraph()
    graph.add_node("B", fn=node_test_fn2, outputs=["switch"])
    graph.add_node("C", fn=node_test_fn4, outputs=["final1", "final2"])
    graph.add_dependency("B", "C")
    inputs = {
        "B": {
            "arg": 1
        }
    }

    result = graph.evaluate(inputs=inputs)
    assert "B" not in result
    assert result["C"]["final1"] == 0.0
    assert result["C"]["final2"] == 1.0

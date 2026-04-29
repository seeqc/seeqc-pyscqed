import pytest

import numpy as np

from pyscqed.evaluation_graph import EvaluationGraph


def test_graph_creation():
    def test_fn1(input: int):
        return input

    def test_fn2(arg: int) -> bool:
        return arg == 1

    graph = EvaluationGraph()
    graph.add_node("A", fn=test_fn1, outputs=["arg"])
    graph.add_node("B", fn=test_fn2, outputs=["final"])
    graph.add_dependency("A", "B", preserve_source_outputs=True)
    inputs = {
        "A": {
            "input": 1
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result["A"]["arg"] == 1
    assert result["B"]["final"] == True

    inputs = {
        "A": {
            "input": 0
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result["A"]["arg"] == 0
    assert result["B"]["final"] == False

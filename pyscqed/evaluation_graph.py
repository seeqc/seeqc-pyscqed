"""Defines the DAG structure for evaluations."""
from typing import Callable, Any
import networkx as nx


class EvaluationGraph:
    def __init__(self):
        self._graph = nx.DiGraph()

    def add_node(self, name: str, fn: Callable, inputs: set[str]):
        self._graph.add_node(name, node_fn=fn, node_inputs=inputs)

    def add_dependency(self, source_node: str, target_node: str, preserve_source_outputs: bool = False):
        self._graph.add_edge(source_node, target_node)

    def evaluate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Get the start nodes
        start_nodes = {node for node in self._graph.nodes if self._graph.in_degree(node) == 0}
        data = {
            node: self._graph.nodes[node]["node_fn"](**inputs[node])
            for node in start_nodes
        }
        new_data = {}
        for node, inputs in data.items():
            node_iter = self._graph.successors(node)
            new_data.update({
                node: self._graph.nodes[node]["node_fn"](inputs)
                for node in node_iter
            })
        data.update(new_data)
        #print(data)
        return data

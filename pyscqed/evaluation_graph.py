"""Defines the DAG structure for evaluations."""
from typing import Callable, Any
import networkx as nx


def _define_outputs(fn: Callable, keys: list[str]) -> Callable:
    """Decorator that maps function outputs to user defined keys."""
    def inner(*args, **kwargs) -> dict[str, Any]:
        outputs = fn(*args, **kwargs)
        if not isinstance(outputs, tuple):
            if len(keys) != 1:
                raise TypeError("A singular function output must map to a single key.")
            return {keys[0]: outputs}
        formatted = {k: v for k, v in zip(keys, outputs)}
        return formatted
    return inner


class EvaluationGraph:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._silent_nodes: set[str] = set()

    def add_node(self, name: str, fn: Callable, outputs: list[str]):
        self._graph.add_node(name, node_fn=fn, node_outputs=outputs)

    def add_dependency(self, source_node: str, target_node: str, preserve_source_outputs: bool = False):
        self._graph.add_edge(source_node, target_node)
        if preserve_source_outputs:
            self._silent_nodes.add(source_node)

    def evaluate(self, inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        # Get the start nodes
        start_nodes = {node for node in self._graph.nodes if self._graph.in_degree(node) == 0}

        # Produce the outputs of the start nodes
        data = self._get_node_outputs(start_nodes, inputs)

        # Get next set of nodes
        next_nodes = set()
        next_inputs = {}
        for node in data:
            node_iter = self._graph.successors(node)
            for inode in node_iter:
                next_nodes.add(inode)

                # The outputs of the previous function become the inputs of the next
                next_inputs[inode] = data[node]

        # Produce the next outputs
        data.update(self._get_node_outputs(next_nodes, next_inputs))
        return data

    def _get_node_outputs(self, nodes: set[int], inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        data = {}
        for node in nodes:
            original_callable = self._graph.nodes[node]["node_fn"]
            user_defined_outputs = self._graph.nodes[node]["node_outputs"]
            wrapped_callable = _define_outputs(
                original_callable,
                user_defined_outputs
            )
            data[node] = wrapped_callable(**inputs[node])
        return data

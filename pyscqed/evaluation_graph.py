"""Defines the DAG structure for evaluations."""
from typing import Callable, Any, TypeAlias
import networkx as nx


NodeIOData: TypeAlias = dict[str, dict[str, Any]]


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

    def addNode(self, name: str, fn: Callable, outputs: list[str]):
        self._graph.add_node(name, node_fn=fn, node_outputs=outputs)

    def addDependency(self, source_node: str, target_node: str, preserve_source_outputs: bool = False):
        self._graph.add_edge(source_node, target_node)
        if not preserve_source_outputs:
            self._silent_nodes.add(source_node)

    def evaluate(self, inputs: NodeIOData) -> dict[str, Any]:
        # Check the graph is a dag
        if not nx.is_directed_acyclic_graph(self._graph):
            raise TypeError("The evaluation graph structure is not a DAG.")

        # Get the start nodes
        next_nodes = self._get_start_nodes()
        next_inputs = inputs
        data = {}
        while len(next_nodes) > 0:
            data.update(self._get_node_outputs(next_nodes, next_inputs))
            next_nodes, next_inputs = self._get_next_nodes_and_inputs(next_nodes, data)

        # TODO: The silent node data should ideally be dropped in the loop
        for node in self._silent_nodes:
            del data[node]
        return data

    def getDefaultInputs(self) -> NodeIOData:
        return {node: {} for node in self._get_start_nodes()}

    def _get_start_nodes(self) -> list[str]:
        return [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]

    def _get_node_outputs(self, nodes: list[str], inputs: NodeIOData) -> dict[str, Any]:
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

    def _get_next_nodes_and_inputs(self, start_nodes: list[str], outputs: NodeIOData) -> tuple[list[str], NodeIOData]:
        next_nodes = []
        next_inputs = {}
        for node in start_nodes:
            node_iter = self._graph.successors(node)
            for inode in node_iter:
                next_nodes.append(inode)
            
                # The outputs of the previous function become the inputs of the next
                next_inputs[inode] = outputs[node]
        return next_nodes, next_inputs

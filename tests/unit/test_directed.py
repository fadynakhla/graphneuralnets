import pytest
import torch
import networkx as nx

from graphneuralnets.message_passing import graph_tensor


DIRECTED = torch.tensor(
    [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ],
    dtype=torch.int,
)
REVERSED = DIRECTED.transpose(dim0=-2, dim1=-1)

NX_GRAPH = nx.from_numpy_array(DIRECTED.numpy())


def graphs_equal(g1: nx.Graph, g2: nx.Graph) -> bool:
    return (set(g1.nodes()) == set(g2.nodes())) and (set(g1.edges()) == set(g2.edges()))


@pytest.mark.parametrize("adjacency, backward", [(DIRECTED, False), (REVERSED, True)])
def test_directed_nx_graph_construction(
    adjacency: graph_tensor.SquareIntTensor, backward: bool
) -> None:
    gt = graph_tensor.GraphTensor(adjacency, backward)
    assert graphs_equal(gt.to_networkx_graph(), NX_GRAPH)


@pytest.mark.parametrize("adjacency, reversed_adjacency", [(DIRECTED, REVERSED)])
def test_equivalent_normalization(
    adjacency: graph_tensor.SquareIntTensor,
    reversed_adjacency: graph_tensor.SquareIntTensor,
) -> None:
    gt = graph_tensor.GraphTensor(adjacency)
    rev_gt = graph_tensor.GraphTensor(reversed_adjacency, backward_convention=True)
    assert torch.equal(
        gt.normalized_adjacency, rev_gt.normalized_adjacency.transpose(dim0=-1, dim1=-2)
    )
    assert torch.equal(
        gt.normalized_conv_adjacency,
        rev_gt.normalized_conv_adjacency.transpose(dim0=-1, dim1=-2),
    )

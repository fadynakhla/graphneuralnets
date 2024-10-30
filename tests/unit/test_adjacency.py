import torch
import pytest

from graphneuralnets.message_passing import graph_tensor


ADJACENCY = torch.tensor(
    [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=torch.int,
)
DEGREE_DIAG = torch.tensor([1, 2, 3, 1, 1], dtype=torch.int)
FEATS = torch.arange(ADJACENCY.shape[0], dtype=torch.float) + 1
H_AVG = torch.Tensor([2.0, 2.0, 11 / 3, 3.0, 3.0])


@pytest.mark.parametrize("adjacency, degree", [(ADJACENCY, DEGREE_DIAG)])
def test_degree_from_adjacency(
    adjacency: graph_tensor.SquareIntTensor, degree: torch.Tensor
) -> None:
    graph = graph_tensor.GraphTensor(adjacency)
    assert torch.equal(torch.diag(graph.in_degree), degree)


@pytest.mark.parametrize("adjacency, feats, result", [(ADJACENCY, FEATS, H_AVG)])
def test_avg_feats(
    adjacency: graph_tensor.SquareIntTensor, feats: torch.Tensor, result: torch.Tensor
) -> None:
    graph = graph_tensor.GraphTensor(adjacency)
    calculated_result = graph.left_normalized_adjacency @ feats
    print(calculated_result)
    assert torch.all(torch.isclose(calculated_result, result))


@pytest.mark.parametrize("adjacency", [ADJACENCY])
def test_networkx_conversion(
    adjacency: graph_tensor.SquareIntTensor
) -> None:
    graph = graph_tensor.GraphTensor(adjacency)
    nx_graph = graph.to_networkx_graph()
    new_graph = graph_tensor.GraphTensor.from_networkx_graph(nx_graph)
    assert torch.equal(new_graph.adjacency, adjacency)

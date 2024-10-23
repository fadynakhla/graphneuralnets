from typing import Literal
import numpy as np
from numpy import typing as npt
import pytest

from graphneuralnets.message_passing import graph_matrix


ADJACENCY = np.array(
    [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
)
DEGREE_DIAG = np.array([1, 2, 3, 1, 1])


@pytest.mark.parametrize(
    "adjacency, degree", [(ADJACENCY, DEGREE_DIAG)]
)
def test_degree_from_adjacency(adjacency: npt.NDArray, degree: npt.NDArray) -> None:
    graph = graph_matrix.GraphMatrix[np.uint](adjacency)
    assert np.array_equal(np.diag(graph.degree_matrix), degree)

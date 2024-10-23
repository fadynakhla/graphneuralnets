from typing import Generic, TypeVar, TypeVarTuple

import numpy as np
from numpy import typing as npt


SquareDim = TypeVar("SquareDim", bound=int)


class GraphMatrix(Generic[SquareDim]):
    def __init__(
        self, adjacency_matrix: npt.NDArray[tuple[SquareDim, SquareDim] :]
    ) -> None:
        super().__init__()
        self.adjacency_matrix = adjacency_matrix

    @classmethod
    def to_degree_matrix(
        cls, adjacency_matrix: npt.NDArray[tuple[SquareDim, SquareDim] :]
    ) -> npt.NDArray[tuple[SquareDim, SquareDim] :]:
        degrees = adjacency_matrix.sum(axis=-1)
        degree_matrix = np.zeros_like(adjacency_matrix)
        np.einsum("...ii->...i", degree_matrix)[:] = degrees
        return degree_matrix

    @classmethod
    def to_convolution_form(
        cls, matrix: npt.NDArray[tuple[SquareDim, SquareDim] :]
    ) -> npt.NDArray[tuple[SquareDim, SquareDim] :]:
        identity = np.eye(matrix.shape[-1], dtype=matrix.dtype)
        return matrix + np.broadcast_to(identity, matrix.shape)

    @property
    def undirected_graph(self) -> "GraphMatrix":
        symmetric_adjacency = np.logical_or(
            self.adjacency_matrix, self.adjacency_matrix.transpose((-2, -1))
        )
        return GraphMatrix(symmetric_adjacency)

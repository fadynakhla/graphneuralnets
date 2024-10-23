from typing import Generic, TypeVar, TypeVarTuple

import numpy as np
from numpy import typing as npt


SquareDim = TypeVar("SquareDim", bound=int)


class GraphMatrix(Generic[SquareDim]):
    def __init__(self, adjacency_matrix: npt.NDArray[tuple[SquareDim, SquareDim]: np.uint]) -> None:
        super().__init__()
        self.adjacency_matrix = adjacency_matrix
        self.degree_matrix = GraphMatrix.to_degree_matrix(self.adjacency_matrix)

    @classmethod
    def to_degree_matrix(cls, adjacency_matrix: npt.NDArray[tuple[SquareDim, SquareDim]: np.uint]) -> npt.NDArray[tuple[SquareDim, SquareDim]: np.uint]:
        degrees = adjacency_matrix.sum(axis=-1)
        degree_matrix = np.zeros_like(adjacency_matrix)
        np.einsum("...ii->...i", degree_matrix)[:] = degrees
        return degree_matrix

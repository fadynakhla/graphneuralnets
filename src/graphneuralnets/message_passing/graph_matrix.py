from typing import Generic, TypeVar, TypeVarTuple

import numpy as np
from numpy import typing as npt
from scipy import linalg as scp_linalg


SquareDim = TypeVar("SquareDim", bound=int)
UIntDType = TypeVar("UIntDType", bound=np.uint, covariant=True)


class GraphMatrix(Generic[UIntDType]):
    def __init__(
        self, adjacency_matrix: npt.NDArray[UIntDType]
    ) -> None:
        super().__init__()
        self.adjacency_matrix = adjacency_matrix

    @property
    def normalized_conv_adjacency(self) -> npt.NDArray[UIntDType]:
        conv_adjacency = self.to_convolution_form(self.adjacency_matrix)
        conv_degree = self.to_degree_matrix(conv_adjacency)
        inv_root_degree = self.inverse_root(conv_degree)
        normalized_conv_adjacency = inv_root_degree @ conv_adjacency @ inv_root_degree
        return normalized_conv_adjacency.astype(self.adjacency_matrix.dtype)

    @property
    def degree_matrix(self) -> npt.NDArray[UIntDType]:
        return self.to_degree_matrix(self.adjacency_matrix)

    @property
    def undirected_graph(self) -> "GraphMatrix":
        symmetric_adjacency = np.logical_or(
            self.adjacency_matrix, self.adjacency_matrix.transpose((-2, -1))
        )
        return GraphMatrix(symmetric_adjacency)

    @classmethod
    def inverse_root(cls, matrix: npt.NDArray[UIntDType]) -> npt.NDArray[UIntDType]:
        return np.linalg.inv(scp_linalg.sqrtm(matrix))

    @classmethod
    def to_degree_matrix(
        cls, adjacency_matrix: npt.NDArray[UIntDType]
    ) -> npt.NDArray[UIntDType]:
        degrees = adjacency_matrix.sum(axis=-1)
        degree_matrix = np.zeros_like(adjacency_matrix)
        np.einsum("...ii->...i", degree_matrix)[:] = degrees
        return degree_matrix

    @classmethod
    def to_convolution_form(
        cls, matrix: npt.NDArray[UIntDType]
    ) -> npt.NDArray[UIntDType]:
        identity = np.eye(matrix.shape[-1], dtype=matrix.dtype)
        conv_form = np.add(matrix, np.broadcast_to(identity, matrix.shape))
        return conv_form

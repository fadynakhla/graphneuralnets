import jaxtyping as jt
import networkx as nx
import torch


SquareIntTensor = jt.Int[torch.Tensor, "N N"]
SquareFloatTensor = jt.Float[torch.Tensor, "N N"]


class GraphTensor:
    def __init__(self, adjacency: SquareIntTensor) -> None:
        self.adjacency = adjacency

    @property
    def degree(self) -> SquareIntTensor:
        return self.to_degree(self.adjacency)

    @property
    def left_normalized_adjacency(self) -> SquareFloatTensor:
        return self.left_side_normalize(self.adjacency)

    @property
    def normalized_adjacency(self) -> SquareFloatTensor:
        return self.normalize(self.adjacency)

    @property
    def normalized_conv_adjacency(self) -> SquareFloatTensor:
        conv_adjacency = self.to_convolution_form(self.adjacency)
        return self.normalize(conv_adjacency)

    @property
    def undirected_graph(self) -> "GraphTensor":
        symmetric_adjacency = torch.logical_or(
            self.adjacency, self.adjacency.transpose(dim0=-2, dim1=-1)
        )
        return GraphTensor(symmetric_adjacency)

    def to_networkx_graph(self) -> nx.Graph:
        return nx.from_numpy_array(self.adjacency.numpy())

    @classmethod
    def to_degree(cls, adjacency: SquareIntTensor) -> SquareIntTensor:
        # This is inefficient as we are allocating the majority of the
        # tensor's memory to storing zeros.
        # TODO: rewrite to use a (..., N) tensor representing the
        # diagonal.
        degree_diag = adjacency.sum(dim=-1)
        degree = torch.zeros_like(adjacency)
        torch.einsum("...ii->...i", degree)[:] = degree_diag
        return degree

    @classmethod
    def to_convolution_form(cls, tensor: SquareIntTensor) -> SquareIntTensor:
        identity = torch.eye(tensor.shape[-1], dtype=tensor.dtype)
        return tensor + identity.expand_as(tensor)

    @classmethod
    def inverse_root(cls, tensor: SquareIntTensor) -> SquareFloatTensor:
        return cls.invert_pos_diagonal_tensor(torch.sqrt(tensor))

    @classmethod
    def invert_pos_diagonal_tensor(cls, tensor: SquareFloatTensor) -> SquareFloatTensor:
        inv = 1.0 / tensor
        inv[torch.isinf(inv)] = 0.0
        return inv

    @classmethod
    def normalize(cls, adjacency_tensor: SquareIntTensor) -> SquareFloatTensor:
        degree = cls.to_degree(adjacency_tensor)
        inv_root_degree = cls.inverse_root(degree)
        return inv_root_degree @ adjacency_tensor.type(inv_root_degree.dtype) @ inv_root_degree

    @classmethod
    def left_side_normalize(cls, adjacency_tensor: SquareIntTensor) -> SquareFloatTensor:
        degree = cls.to_degree(adjacency_tensor)
        inv_degree = cls.invert_pos_diagonal_tensor(degree)
        return inv_degree @ adjacency_tensor.type(inv_degree.dtype)

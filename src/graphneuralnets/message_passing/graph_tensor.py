import jaxtyping as jt
import networkx as nx
import torch


SquareIntTensor = jt.Int[torch.Tensor, "N N"]
SquareFloatTensor = jt.Float[torch.Tensor, "N N"]


class GraphTensor:
    """A tensor based representation of a (possibly directed) graph in
    torch.

    Expects an adjacency matrix upon initialization and provides methods
    for calculating several other useful matrices such as the degree
    matrices, normalized adjacency matrices, etc. Here we use the
    convention that each row (dim -1) represents the in edges to the
    node sharing it's index, i.e. A_ij > 0 if there is an edge from node
    j to node i. This means that the standard matrix multiplication
    Ax, where A is the adjacency and x is a vector of features, sums the
    features of the incomming nodes.
    """

    def __init__(self, adjacency: SquareIntTensor) -> None:
        self.adjacency = adjacency

    @property
    def in_degree(self) -> SquareIntTensor:
        return self.to_in_degree(self.adjacency)

    @property
    def out_degree(self) -> SquareIntTensor:
        return self.to_out_degree(self.adjacency)

    @property
    def normalized_adjacency(self) -> SquareFloatTensor:
        return self.normalize(self.adjacency)

    @property
    def normalized_conv_adjacency(self) -> SquareFloatTensor:
        conv_adjacency = self.to_convolution_form(self.adjacency)
        return self.normalize(conv_adjacency)

    @property
    def left_normalized_adjacency(self) -> SquareFloatTensor:
        return self.left_side_normalize(self.adjacency)

    @property
    def undirected_graph(self) -> "GraphTensor":
        symmetric_adjacency = torch.logical_or(
            self.adjacency, self.adjacency.transpose(dim0=-2, dim1=-1)
        )
        return GraphTensor(symmetric_adjacency)

    def to_networkx_graph(self) -> nx.Graph:
        return nx.from_numpy_array(self.adjacency.numpy())

    @classmethod
    def from_networkx_graph(cls, graph: nx.Graph) -> "GraphTensor":
        adjacency = nx.to_numpy_array(graph)
        adjacency_tensor = torch.tensor(adjacency, dtype=torch.int)
        return cls(adjacency_tensor)

    @classmethod
    def to_in_degree(cls, adjacency: SquareIntTensor) -> SquareIntTensor:
        degree_diag = adjacency.sum(dim=-1)
        degree = torch.diag_embed(degree_diag)
        return degree

    @classmethod
    def to_out_degree(cls, adjacency: SquareIntTensor) -> SquareIntTensor:
        degree_diag = adjacency.sum(dim=-2)
        degree = torch.diag_embed(degree_diag)
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
        inv_root_in_degree = cls.inverse_root(cls.to_in_degree(adjacency_tensor))
        inv_root_out_degree = cls.inverse_root(cls.to_out_degree(adjacency_tensor))
        return (
            inv_root_in_degree
            @ adjacency_tensor.type(inv_root_in_degree.dtype)
            @ inv_root_out_degree
        )

    @classmethod
    def left_side_normalize(
        cls, adjacency_tensor: SquareIntTensor
    ) -> SquareFloatTensor:
        inv_degree = cls.invert_pos_diagonal_tensor(cls.to_in_degree(adjacency_tensor))
        return inv_degree @ adjacency_tensor.type(inv_degree.dtype)

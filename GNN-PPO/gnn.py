import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.utils import scatter


class GINEncoder(nn.Module):
    """
    GIN encoder for RCPSP activity graphs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        first_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers.append(
            GINConv(first_mlp, train_eps=True)
        )

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.layers.append(
                GINConv(mlp, train_eps=True)
            )

    def forward(self, x, edge_index, batch=None):

        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.relu(x)

        node_embeddings = x

        if batch is None:
            graph_embedding = node_embeddings.mean(dim=0)
        else:
            graph_embedding = scatter(
                node_embeddings,
                batch,
                dim=0,
                reduce="mean",
            )

        return node_embeddings, graph_embedding
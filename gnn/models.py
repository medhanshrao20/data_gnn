import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear projection weight
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention parameters (split into two vectors for source and target features)
        self.a_src = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_dst = nn.Parameter(torch.zeros(size=(out_features, 1)))
        
        # Initialization
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: shape (N, in_features)
        # adj: shape (N, N)
        N = h.size(0)
        
        # Project node features
        h_prime = self.W(h) # shape (N, out_features)

        # Compute attention inputs
        # f_src: (N, 1), f_dst: (N, 1)
        f_src = torch.matmul(h_prime, self.a_src)
        f_dst = torch.matmul(h_prime, self.a_dst)
        
        # Broadcast sum to get pairwise attention logits: (N, N)
        # e_ij = LeakyReLU(a_src * h_i + a_dst * h_j)
        logits = f_src + f_dst.t()
        signed_edge_bias = torch.sign(adj) * torch.log1p(torch.abs(adj))
        attn_logits = self.leakyrelu(logits + signed_edge_bias)

        # Mask attention using adjacency matrix
        # GAT adds self loops to ensure every node attends to itself and avoids division by zero
        mask_adj = torch.abs(adj) + torch.eye(N, device=adj.device)
        
        # Set attention to very low values where there is no edge
        zero_vec = -9e15 * torch.ones_like(attn_logits)
        attention = torch.where(mask_adj > 0, attn_logits, zero_vec)
        
        # Softmax over neighbors
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate neighbor features
        h_out = torch.matmul(attention, h_prime) # shape (N, out_features)
        return h_out


class GATModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float = 0.1):
        super(GATModel, self).__init__()
        self.layer1 = GATLayer(in_features, hidden_dim, dropout=dropout)
        self.layer2 = GATLayer(hidden_dim, out_features, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return x


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, aggregator_type: str = "mean"):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator_type = aggregator_type  # 'mean' or 'max'
        
        # Linear layer for self and neighborhood features combined
        self.W = nn.Linear(in_features * 2, out_features)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: shape (N, in_features)
        # adj: shape (N, N)
        N = h.size(0)
        
        # Compute degrees (sum of row in adj)
        deg = torch.sum(torch.abs(adj), dim=1, keepdim=True)
        # Avoid division by zero
        deg = torch.clamp(deg, min=1.0)
        
        if self.aggregator_type == "mean":
            # h_neigh = (D^-1 @ A) @ h
            h_neigh = torch.matmul(adj / deg, h)
        elif self.aggregator_type == "max":
            # For each node, neighborhood features are max of neighbor features
            # adj is shape (N, N). We can expand h to shape (N, N, in_features)
            # Mask out non-neighbors and take max
            adj_expanded = adj.unsqueeze(2) # (N, N, 1)
            h_expanded = h.unsqueeze(0).expand(N, -1, -1) # (N, N, in_features)
            weighted_h = h_expanded * adj_expanded
            # set non-neighbors to a small value
            masked_h = torch.where(torch.abs(adj_expanded) > 0, weighted_h, torch.tensor(-9e15, device=h.device))
            h_neigh, _ = torch.max(masked_h, dim=1) # (N, in_features)
            # Replace -9e15 with 0 if a node has no neighbors
            h_neigh = torch.where(h_neigh < -1e10, torch.zeros_like(h_neigh), h_neigh)
        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")

        # Concatenate self features and neighborhood aggregated features
        h_concat = torch.cat([h, h_neigh], dim=1) # (N, 2 * in_features)
        
        # Project to output space
        h_out = self.W(h_concat)
        return h_out


class GraphSAGEModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, aggregator_type: str = "mean"):
        super(GraphSAGEModel, self).__init__()
        self.layer1 = GraphSAGELayer(in_features, hidden_dim, aggregator_type=aggregator_type)
        self.layer2 = GraphSAGELayer(hidden_dim, out_features, aggregator_type=aggregator_type)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x, adj)
        x = F.relu(x)
        x = self.layer2(x, adj)
        return x

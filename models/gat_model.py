"""Graph Attention Network model for Sybil detection."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SybilGAT(torch.nn.Module):
    """
    Graph Attention Network for Sybil node classification.
    
    Architecture:
        - 2-layer GAT with multi-head attention
        - Dropout for regularization
        - ELU activation between layers
        - Log-softmax output for classification
    """
    
    def __init__(self, num_features, hidden_channels, num_classes, heads=8):
        """
        Initialize the SybilGAT model.
        
        Args:
            num_features: Number of input features per node
            hidden_channels: Number of hidden units
            num_classes: Number of output classes (2 for binary classification)
            heads: Number of attention heads
        """
        super(SybilGAT, self).__init__()
        self.conv1 = GATConv(
            num_features, 
            hidden_channels, 
            heads=heads, 
            dropout=0.6
        )
        self.conv2 = GATConv(
            hidden_channels * heads, 
            num_classes, 
            heads=1, 
            concat=False, 
            dropout=0.6
        )

    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Log-softmax probabilities [num_nodes, num_classes]
        """
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

"""Graph Attention Network model for Sybil detection.

Supports two modes:
1. Classification: Full forward pass with log_softmax output
2. Embedding: Return intermediate representations for clustering

Architecture adapted from colab-code/train.py (SybilGAT class, lines 247-271)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple


class SybilGAT(torch.nn.Module):
    """
    Graph Attention Network for Sybil detection.
    
    Architecture:
        Layer 1: GAT (num_features -> hidden_channels * heads)
        Layer 2: GAT (hidden_channels * heads -> hidden_channels * 2)
        Layer 3: GAT (hidden_channels * 2 -> num_classes)
    
    Args:
        num_features: Number of input features per node
        hidden_channels: Hidden layer dimension (default: 32)
        num_classes: Number of output classes (default: 2)
        heads: Number of attention heads for first layer (default: 4)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self, 
        num_features: int, 
        hidden_channels: int = 32, 
        num_classes: int = 2, 
        heads: int = 4,
        dropout: float = 0.3
    ):
        super(SybilGAT, self).__init__()
        
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        # Layer 1: Feature Extraction
        # Input: num_features -> Output: hidden_channels * heads
        self.conv1 = GATConv(
            num_features, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout
        )
        
        # Layer 2: Refinement
        # Input: hidden_channels * heads -> Output: hidden_channels * 2
        self.conv2 = GATConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=2, 
            dropout=dropout
        )
        
        # Layer 3: Classification
        # Input: hidden_channels * 2 -> Output: num_classes
        self.conv3 = GATConv(
            hidden_channels * 2, 
            num_classes, 
            heads=1, 
            concat=False, 
            dropout=dropout
        )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Node features tensor [num_nodes, num_features]
            edge_index: Edge indices tensor [2, num_edges]
            return_embeddings: If True, return Layer 2 embeddings instead of classification
            
        Returns:
            If return_embeddings=False: Log-softmax probabilities [num_nodes, num_classes]
            If return_embeddings=True: Node embeddings [num_nodes, hidden_channels * 2]
        """
        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Return embeddings if requested (for clustering)
        if return_embeddings:
            return x
        
        # Layer 3 (Classification)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract node embeddings from Layer 2.
        
        This is useful for clustering tasks where we need
        intermediate representations instead of class predictions.
        
        Args:
            x: Node features tensor [num_nodes, num_features]
            edge_index: Edge indices tensor [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, hidden_channels * 2]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index, return_embeddings=True)
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings from Layer 2."""
        return self.hidden_channels * 2


class GATEncoder(torch.nn.Module):
    """
    GAT Encoder for Graph Auto-Encoder (GAE).
    
    Used for unsupervised learning to generate node embeddings
    without labels. Architecture adapted from colab-code/gae.py (lines 349-362)
    
    Args:
        in_channels: Number of input features
        out_channels: Embedding dimension (default: 64)
        heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.3
    ):
        super(GATEncoder, self).__init__()
        
        self.dropout = dropout
        
        # Layer 1: Multi-head attention for diverse information
        self.conv1 = GATConv(
            in_channels, 
            32, 
            heads=heads, 
            dropout=dropout
        )
        
        # Layer 2: Compress to final embedding
        self.conv2 = GATConv(
            32 * heads, 
            out_channels, 
            heads=1, 
            concat=False, 
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate node embeddings.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
"""Graph Attention Network model for Sybil detection."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SybilGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=4):
        super(SybilGAT, self).__init__()
        
        # Layer 1: Feature Extraction
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=0.3)
        
        # Layer 2: Refinement
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=2, dropout=0.3)
        
        # Layer 3: Classification (Lớp mới thêm)
        self.conv3 = GATConv(hidden_channels * 2, num_classes, heads=1, concat=False, dropout=0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

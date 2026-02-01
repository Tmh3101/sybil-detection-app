"""Network visualization utilities for Sybil detection."""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd
import torch
from typing import Dict, List, Optional, Any


def visualize_prediction_graph(
    node_info: Dict[str, Any],
    new_edge_index: torch.Tensor,
    new_edge_types: List[str],
    df_ref: pd.DataFrame,
    prediction_result: Dict[str, Any]
) -> Optional[Figure]:
    """
    Create a multi-layer network visualization for Sybil analysis.
    
    Args:
        node_info: Dictionary containing target node information.
        new_edge_index: Tensor of edge indices [2, num_edges].
        new_edge_types: List of edge type labels corresponding to edges.
        df_ref: Reference DataFrame containing node information.
        prediction_result: Dictionary with prediction results.
        
    Returns:
        matplotlib Figure object, or None if node is isolated.
    """
    # Check for isolated node
    if new_edge_index.numel() == 0:
        return None

    # Initialize graph
    G = nx.Graph()
    edges = new_edge_index.cpu().numpy()

    # Determine target node
    target_idx = len(df_ref)
    target_label = node_info.get('handle', 'Target')

    # Color based on prediction
    is_sybil = prediction_result['prediction'] == 'SYBIL'
    target_color = '#dc2626' if is_sybil else '#16a34a'  # Red / Green

    # Add target node
    G.add_node(target_idx, label=target_label, node_type='target', color=target_color)

    # Process edges with priority handling
    priority_map = {'co_owner': 3, 'similarity': 2, 'social': 1}

    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        etype = new_edge_types[i]

        # Determine neighbor index
        neighbor_idx = v if u == target_idx else u

        # Add neighbor node if not exists
        if not G.has_node(neighbor_idx):
            ref_row = df_ref.iloc[neighbor_idx]
            G.add_node(
                neighbor_idx,
                label=ref_row['handle'],
                node_type='neighbor',
                color='#e5e7eb'  # Light gray
            )

        # Add or update edge (prioritize higher risk types)
        if G.has_edge(target_idx, neighbor_idx):
            current_type = G[target_idx][neighbor_idx]['edge_type']
            if priority_map.get(etype, 0) > priority_map.get(current_type, 0):
                G.add_edge(target_idx, neighbor_idx, edge_type=etype)
        else:
            G.add_edge(target_idx, neighbor_idx, edge_type=etype)

    # Compute layout
    pos = nx.spring_layout(G, k=0.4, seed=42)

    # Create figure with clean styling
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    # Remove all spines and ticks for clean look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw edges by layer

    # Layer 1: Social connections (subtle gray)
    edges_social = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'social']
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=edges_social,
        width=1.0, alpha=0.3, edge_color='#9ca3af'
    )

    # Layer 2: Similarity edges (purple, dotted)
    edges_sim = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'similarity']
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=edges_sim,
        width=1.5, alpha=0.7, edge_color='#7c3aed', style='dotted',
        connectionstyle='arc3,rad=0.15'
    )

    # Layer 3: Co-owner edges (red, dashed - highest risk)
    edges_co = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'co_owner']
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=edges_co,
        width=2.0, alpha=0.9, edge_color='#dc2626', style='dashed',
        connectionstyle='arc3,rad=0.3'
    )

    # Draw nodes
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    node_sizes = [1800 if data['node_type'] == 'target' else 600 for _, data in G.nodes(data=True)]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='#374151',
        linewidths=1.0
    )

    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos, ax=ax, labels=labels,
        font_size=8,
        font_family='sans-serif',
        font_weight='medium',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.2)
    )

    # Create minimal legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target',
               markerfacecolor=target_color, markersize=12, markeredgecolor='#374151'),
        Line2D([0], [0], marker='o', color='w', label='Reference',
               markerfacecolor='#e5e7eb', markersize=9, markeredgecolor='#374151'),
        Line2D([0], [0], color='#dc2626', lw=2, linestyle='dashed', label='Co-owner'),
        Line2D([0], [0], color='#7c3aed', lw=1.5, linestyle='dotted', label='Similarity'),
        Line2D([0], [0], color='#9ca3af', lw=1, label='Social'),
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=9,
        frameon=False,
        labelspacing=0.8
    )

    plt.tight_layout()
    return fig

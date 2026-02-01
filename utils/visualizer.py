"""Network visualization utilities for Sybil detection.

Supports two visualization modes:
1. Static (Matplotlib) - For reports and fallback
2. Interactive (PyVis) - For exploration and analysis

Edge Categories:
- FOLLOW: Directed (Layer 1)
- INTERACT: Directed - UPVOTE, COMMENT, QUOTE, MIRROR, COLLECT, TIP (Layer 2)
- CO-OWNER: Undirected (Layer 3)
- SIMILARITY: Undirected - SAME_AVATAR, FUZZY_HANDLE, SIM_BIO, CLOSE_CREATION_TIME (Layer 4)
"""

import os
import tempfile
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd
import torch
from typing import Dict, List, Optional, Any, Tuple

# Debug storage for Streamlit display
_debug_messages: List[str] = []

# Edge type categories
FOLLOW_TYPES = {'FOLLOW'}
INTERACT_TYPES = {'UPVOTE', 'COMMENT', 'QUOTE', 'MIRROR', 'COLLECT', 'TIP', 'REACTION'}
CO_OWNER_TYPES = {'CO-OWNER'}
SIMILARITY_TYPES = {'SAME_AVATAR', 'FUZZY_HANDLE', 'SIM_BIO', 'CLOSE_CREATION_TIME'}


def get_edge_category(edge_type: str) -> str:
    if edge_type in FOLLOW_TYPES:
        return 'follow'
    elif edge_type in INTERACT_TYPES:
        return 'interact'
    elif edge_type in CO_OWNER_TYPES:
        return 'co_owner'
    elif edge_type in SIMILARITY_TYPES:
        return 'similarity'
    else:
        return 'interact'  # Default to interact


def get_debug_messages() -> List[str]:
    return _debug_messages.copy()


def clear_debug_messages() -> None:
    _debug_messages.clear()


def _log(msg: str, level: str = "INFO") -> None:
    _debug_messages.append(f"[{level}] {msg}")


# PyVis import (optional)
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


def _build_analysis_graph(
    node_info: Dict[str, Any],
    new_edge_index: torch.Tensor,
    new_edge_types: List[str],
    new_edge_dirs: List[str],
    df_ref: pd.DataFrame,
    prediction_result: Dict[str, Any],
    ref_labels: Optional[torch.Tensor] = None
) -> Tuple[nx.DiGraph, int, str]:
    """
    Build a NetworkX directed graph from prediction results.
    
    Args:
        node_info: Target node information.
        new_edge_index: Edge indices tensor.
        new_edge_types: Edge type labels.
        new_edge_dirs: Edge directions ('outgoing', 'incoming', 'undirected').
        df_ref: Reference DataFrame.
        prediction_result: Prediction results.
        ref_labels: Labels for reference nodes (0=non-sybil, 1=sybil).
        
    Returns:
        Tuple of (NetworkX DiGraph, target_idx, target_color)
    """
    G = nx.DiGraph()
    edges = new_edge_index.cpu().numpy()
    
    # Determine target node
    target_idx = len(df_ref)
    target_label = node_info.get('handle', 'Target')
    
    # Color based on prediction
    is_sybil = prediction_result['prediction'] == 'SYBIL'
    target_color = '#dc2626' if is_sybil else '#16a34a'
    
    # Add target node with attributes
    G.add_node(
        target_idx,
        label=target_label,
        node_type='target',
        color=target_color,
        is_sybil=is_sybil
    )
    
    # Priority map for edge categories (higher = more important)
    priority_map = {'co_owner': 4, 'similarity': 3, 'follow': 2, 'interact': 1}
    
    # Track edges to handle duplicates
    # Key: (src, dst) for directed, (min, max) for undirected
    # Value: (category, etype, is_undirected)
    edge_data = {}
    
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        etype = new_edge_types[i]
        direction = new_edge_dirs[i] if i < len(new_edge_dirs) else 'outgoing'
        category = get_edge_category(etype)
        
        # Add nodes if not exists
        for node_idx in [src, dst]:
            if node_idx != target_idx and not G.has_node(node_idx):
                ref_row = df_ref.iloc[node_idx]
                handle = ref_row.get('handle', f'Node_{node_idx}')
                if pd.isna(handle) or str(handle) == 'nan':
                    handle = f'Node_{node_idx}'
                
                # Determine if reference node is sybil
                ref_is_sybil = False
                if ref_labels is not None and node_idx < len(ref_labels):
                    ref_is_sybil = ref_labels[node_idx].item() == 1
                
                ref_color = '#dc2626' if ref_is_sybil else '#16a34a'
                
                G.add_node(
                    node_idx,
                    label=str(handle),
                    node_type='reference',
                    color=ref_color,
                    is_sybil=ref_is_sybil
                )
        
        # Determine if edge is undirected
        is_undirected = direction == 'undirected'
        
        # For undirected edges, normalize direction to avoid duplicates
        if is_undirected:
            edge_key = (min(src, dst), max(src, dst))
        else:
            # Only track outgoing edges (skip incoming since they are reverse for GNN)
            if direction == 'incoming':
                continue
            edge_key = (src, dst)
        
        # Update edge if new category has higher priority
        if edge_key in edge_data:
            existing_category = edge_data[edge_key][0]
            if priority_map.get(category, 0) > priority_map.get(existing_category, 0):
                edge_data[edge_key] = (category, etype, is_undirected)
        else:
            edge_data[edge_key] = (category, etype, is_undirected)
    
    # Add edges to graph
    for (src, dst), (category, etype, is_undirected) in edge_data.items():
        G.add_edge(src, dst, edge_category=category, edge_type=etype, undirected=is_undirected)
        # For undirected edges, add reverse edge too
        if is_undirected:
            G.add_edge(dst, src, edge_category=category, edge_type=etype, undirected=is_undirected)
    
    return G, target_idx, target_color


def render_static_graph(
    node_info: Dict[str, Any],
    new_edge_index: torch.Tensor,
    new_edge_types: List[str],
    new_edge_dirs: List[str],
    df_ref: pd.DataFrame,
    prediction_result: Dict[str, Any],
    ref_labels: Optional[torch.Tensor] = None
) -> Optional[Figure]:
    """
    Create a static matplotlib visualization for Sybil analysis.
    
    Args:
        new_edge_dirs: Edge directions ('outgoing', 'incoming', 'undirected')
    """
    if new_edge_index.numel() == 0:
        return None
    
    G, target_idx, target_color = _build_analysis_graph(
        node_info, new_edge_index, new_edge_types, new_edge_dirs, df_ref, prediction_result, ref_labels
    )
    
    # Compute layout
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Create figure with clean styling
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Remove all spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Separate edges by category
    edges_follow = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_category') == 'follow']
    edges_interact = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_category') == 'interact']
    edges_similarity = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_category') == 'similarity']
    edges_co_owner = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_category') == 'co_owner']
    
    # Remove duplicate undirected edges for drawing
    def dedupe_undirected(edges):
        seen = set()
        result = []
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                result.append((u, v))
        return result
    
    edges_similarity = dedupe_undirected(edges_similarity)
    edges_co_owner = dedupe_undirected(edges_co_owner)
    
    # Draw edges by category
    
    # Layer 1: Follow (directed, blue)
    if edges_follow:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edges_follow,
            width=1.5, alpha=0.7, edge_color='#3b82f6',
            arrows=True, arrowsize=15, arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1',
            min_source_margin=15, min_target_margin=15
        )
    
    # Layer 2: Interact (directed, cyan)
    if edges_interact:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edges_interact,
            width=1.2, alpha=0.6, edge_color='#06b6d4',
            arrows=True, arrowsize=12, arrowstyle='-|>',
            connectionstyle='arc3,rad=0.15',
            min_source_margin=15, min_target_margin=15
        )
    
    # Layer 3: Co-owner (undirected, red, dashed)
    if edges_co_owner:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edges_co_owner,
            width=2.5, alpha=0.9, edge_color='#dc2626', style='dashed',
            arrows=False,
            connectionstyle='arc3,rad=0.2'
        )
    
    # Layer 4: Similarity (undirected, purple, dotted)
    if edges_similarity:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edges_similarity,
            width=1.5, alpha=0.7, edge_color='#7c3aed', style='dotted',
            arrows=False,
            connectionstyle='arc3,rad=0.25'
        )
    
    # Draw reference nodes first (smaller)
    ref_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'reference']
    ref_colors = [G.nodes[n]['color'] for n in ref_nodes]
    if ref_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=ref_nodes,
            node_color=ref_colors,
            node_size=600,
            edgecolors='#374151',
            linewidths=1.0,
            alpha=0.85
        )
    
    # Draw target node (larger, with glow effect)
    target_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'target']
    if target_nodes:
        # Outer glow
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=target_nodes,
            node_color=target_color,
            node_size=2500,
            alpha=0.3
        )
        # Main node
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=target_nodes,
            node_color=target_color,
            node_size=1800,
            edgecolors='#ffffff',
            linewidths=3.0
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
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target (Sybil)',
               markerfacecolor='#dc2626', markersize=14, markeredgecolor='#fff', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', label='Target (Non-Sybil)',
               markerfacecolor='#16a34a', markersize=14, markeredgecolor='#fff', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', label='Ref: Sybil',
               markerfacecolor='#dc2626', markersize=9, markeredgecolor='#374151'),
        Line2D([0], [0], marker='o', color='w', label='Ref: Non-Sybil',
               markerfacecolor='#16a34a', markersize=9, markeredgecolor='#374151'),
        Line2D([0], [0], color='#3b82f6', lw=2, marker='>', markersize=6, label='Follow'),
        Line2D([0], [0], color='#06b6d4', lw=1.5, marker='>', markersize=5, label='Interact'),
        Line2D([0], [0], color='#dc2626', lw=2.5, linestyle='dashed', label='Co-owner'),
        Line2D([0], [0], color='#7c3aed', lw=1.5, linestyle='dotted', label='Similarity'),
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=8,
        frameon=False,
        labelspacing=0.7
    )
    
    plt.tight_layout()
    return fig


def render_interactive_graph(
    node_info: Dict[str, Any],
    new_edge_index: torch.Tensor,
    new_edge_types: List[str],
    new_edge_dirs: List[str],
    df_ref: pd.DataFrame,
    prediction_result: Dict[str, Any],
    ref_labels: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None
) -> Optional[str]:
    """
    Create an interactive PyVis visualization for Sybil analysis.
    
    Args:
        new_edge_dirs: Edge directions ('outgoing', 'incoming', 'undirected')
    """
    clear_debug_messages()
    
    if not PYVIS_AVAILABLE:
        _log("PyVis is not available", "ERROR")
        return None
    
    if new_edge_index.numel() == 0:
        _log("No edges to render", "ERROR")
        return None
    
    try:
        G, target_idx, target_color = _build_analysis_graph(
            node_info, new_edge_index, new_edge_types, new_edge_dirs, df_ref, prediction_result, ref_labels
        )
        _log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        _log(f"Error building graph: {e}", "ERROR")
        return None
    
    try:
        # Use directed graph for PyVis
        nt = Network(
            height="550px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,  # Enable directed edges
            cdn_resources='remote'
        )
        
        nt.set_options("""
        {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "font": {"size": 14, "face": "arial"}
            },
            "edges": {
                "smooth": {"type": "curvedCW", "roundness": 0.15},
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}}
            },
            "physics": {
                "enabled": true,
                "stabilization": {"enabled": true, "iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.3,
                    "springLength": 150,
                    "springConstant": 0.05
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true
            }
        }
        """)
    except Exception as e:
        _log(f"Error initializing PyVis: {e}", "ERROR")
        return None
    
    # Add nodes to PyVis
    try:
        for node in G.nodes():
            attrs = G.nodes[node]
            label = attrs.get('label', str(node)[:8])
            node_type = attrs.get('node_type', 'reference')
            color = attrs.get('color', '#e5e7eb')
            is_sybil = attrs.get('is_sybil', False)
            
            if node_type == 'target':
                size = 45
                border_width = 4
                border_color = '#ffffff'
                status = "SYBIL" if is_sybil else "NON-SYBIL"
                title = f"TARGET: {label}\n{status}\nConfidence: {prediction_result['sybil_probability_formatted']}"
            else:
                size = 22
                border_width = 2
                border_color = '#1a1a2e'
                status = "Sybil" if is_sybil else "Non-Sybil"
                title = f"{label}\nReference ({status})"
            
            node_id = str(node)
            
            nt.add_node(
                node_id,
                label=str(label),
                title=title,
                color={
                    'background': color,
                    'border': border_color,
                    'highlight': {'background': color, 'border': '#ffffff'}
                },
                size=size,
                borderWidth=border_width,
                font={'color': 'white', 'size': 12 if node_type == 'target' else 10}
            )
    except Exception as e:
        _log(f"Error adding nodes: {e}", "ERROR")
        return None
    
    # Add edges to PyVis
    try:
        # Track added undirected edges to avoid duplicates
        added_undirected = set()
        
        for u, v, data in G.edges(data=True):
            category = data.get('edge_category', 'interact')
            etype = data.get('edge_type', '')
            is_undirected = data.get('undirected', False)
            
            u_str, v_str = str(u), str(v)
            
            # Skip duplicate undirected edges
            if is_undirected:
                edge_key = (min(u, v), max(u, v))
                if edge_key in added_undirected:
                    continue
                added_undirected.add(edge_key)
            
            if category == 'follow':
                # Follow: Blue, directed
                nt.add_edge(
                    u_str, v_str, 
                    color='#3b82f6', 
                    width=2, 
                    arrows='to',
                    title=f'Follow'
                )
            elif category == 'interact':
                # Interact: Cyan, directed
                nt.add_edge(
                    u_str, v_str, 
                    color='#06b6d4', 
                    width=1.5, 
                    arrows='to',
                    title=f'Interact ({etype})'
                )
            elif category == 'co_owner':
                # Co-owner: Red, undirected, dashed
                nt.add_edge(
                    u_str, v_str, 
                    color='#dc2626', 
                    width=4, 
                    dashes=True,
                    arrows='',  # No arrows for undirected
                    title='Co-owner (Same Wallet)'
                )
            elif category == 'similarity':
                # Similarity: Purple, undirected, dotted
                nt.add_edge(
                    u_str, v_str, 
                    color='#7c3aed', 
                    width=2, 
                    dashes=[5, 5],
                    arrows='',  # No arrows for undirected
                    title=f'Similarity ({etype})'
                )
    except Exception as e:
        _log(f"Error adding edges: {e}", "ERROR")
        return None
    
    # Save graph
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), 'sybil_graph.html')
    
    try:
        nt.save_graph(output_path)
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            _log("Generated HTML file is empty or missing", "ERROR")
            return None
            
        return output_path
    except Exception as e:
        _log(f"Error saving graph: {e}", "ERROR")
        return None


def create_legend_html() -> str:
    """Create an HTML legend for the interactive graph."""
    return """
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        padding: 12px 16px;
        background: rgba(26, 26, 46, 0.95);
        border-radius: 8px;
        margin-top: -8px;
        margin-bottom: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 12px;
        color: #e5e7eb;
    ">
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 16px; height: 16px; border-radius: 50%; background: #dc2626; border: 3px solid #fff;"></div>
            <span>Target (Sybil)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 16px; height: 16px; border-radius: 50%; background: #16a34a; border: 3px solid #fff;"></div>
            <span>Target (Non-Sybil)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: #dc2626; border: 2px solid #1a1a2e;"></div>
            <span>Ref: Sybil</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: #16a34a; border: 2px solid #1a1a2e;"></div>
            <span>Ref: Non-Sybil</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 20px; height: 2px; background: #3b82f6;"></div>
            <span style="margin-left: -4px;">→ Follow</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 20px; height: 1.5px; background: #06b6d4;"></div>
            <span style="margin-left: -4px;">→ Interact</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 20px; height: 3px; background: #dc2626;"></div>
            <span>Co-owner</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 20px; height: 2px; background: #7c3aed;"></div>
            <span>Similarity</span>
        </div>
    </div>
    """


# Keep backward compatibility
def visualize_prediction_graph(
    node_info: Dict[str, Any],
    new_edge_index: torch.Tensor,
    new_edge_types: List[str],
    new_edge_dirs: List[str],
    df_ref: pd.DataFrame,
    prediction_result: Dict[str, Any],
    ref_labels: Optional[torch.Tensor] = None
) -> Optional[Figure]:
    return render_static_graph(
        node_info, new_edge_index, new_edge_types, new_edge_dirs, df_ref, prediction_result, ref_labels
    )

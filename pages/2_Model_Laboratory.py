"""
Model Laboratory - Data & Model Experimentation Workbench

A scientific workbench for:
1. Data ingestion and preprocessing
2. Unsupervised clustering (behavioral grouping)
3. Semi-supervised labeling (rule-based pseudo-labels)
4. Supervised GAT training
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import io

from utils.visualizer import create_legend_html

# Page configuration
st.set_page_config(
    page_title="Model Laboratory - Lens Sybil Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Typography */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Scorecard styling */
    .scorecard {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .scorecard-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .scorecard-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.2;
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(156, 163, 175, 0.3);
    }
    
    /* Status badges */
    .status-success {
        color: #16a34a;
        font-weight: 600;
    }
    .status-warning {
        color: #d97706;
        font-weight: 600;
    }
    .status-error {
        color: #dc2626;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def render_scorecard(label: str, value: str, delta: str = "") -> None:
    """Render a styled scorecard."""
    delta_html = f'<p style="font-size: 0.875rem; color: #64748b; margin-top: 0.25rem;">{delta}</p>' if delta else ''
    st.markdown(f'''
    <div class="scorecard">
        <p class="scorecard-label">{label}</p>
        <p class="scorecard-value">{value}</p>
        {delta_html}
    </div>
    ''', unsafe_allow_html=True)


def validate_dataframes(
    nodes_df: pd.DataFrame, 
    edges_df: pd.DataFrame
) -> tuple:
    """Validate uploaded dataframes."""
    errors = []
    warnings = []
    
    # Check required columns
    if 'profile_id' not in nodes_df.columns:
        errors.append("nodes.csv missing 'profile_id' column")
    
    if 'source' not in edges_df.columns or 'target' not in edges_df.columns:
        errors.append("edges.csv missing 'source' or 'target' columns")
    
    # Check for optional but useful columns
    useful_cols = ['handle', 'trust_score', 'created_on', 'owned_by']
    missing_useful = [c for c in useful_cols if c not in nodes_df.columns]
    if missing_useful:
        warnings.append(f"Optional columns missing: {', '.join(missing_useful)}")
    
    if 'type' not in edges_df.columns:
        warnings.append("edges.csv missing 'type' column - will use default weights")
    
    return errors, warnings


def remove_isolated_nodes(
    nodes_df: pd.DataFrame, 
    edges_df: pd.DataFrame
) -> tuple:
    """Remove nodes with degree 0 (no connections)."""
    # Get all nodes that appear in edges
    connected_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
    
    # Filter nodes
    original_count = len(nodes_df)
    nodes_df = nodes_df[nodes_df['profile_id'].isin(connected_nodes)].copy()
    removed_count = original_count - len(nodes_df)
    
    # Filter edges to only include valid nodes
    valid_nodes = set(nodes_df['profile_id'])
    edges_df = edges_df[
        (edges_df['source'].isin(valid_nodes)) & 
        (edges_df['target'].isin(valid_nodes))
    ].copy()
    
    return nodes_df, edges_df, removed_count


def count_isolated_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> int:
    """Count nodes with no edges."""
    if nodes_df.empty or edges_df.empty:
        return len(nodes_df)
    
    connected_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
    isolated_count = len(nodes_df[~nodes_df['profile_id'].isin(connected_nodes)])
    return isolated_count


def render_data_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, viz_mode: str) -> None:
    """Render network graph visualization."""
    if nodes_df.empty or edges_df.empty:
        st.info("No data to visualize.")
        return
    
    if len(nodes_df) > 500:
        st.info(f"Graph too large ({len(nodes_df)} nodes). Visualization skipped for performance.")
        return
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for _, row in nodes_df.iterrows():
        handle = row.get('handle', str(row['profile_id'])[:8])
        G.add_node(
            row['profile_id'],
            label=handle[:12] if len(str(handle)) > 12 else handle,
            trust_score=row.get('trust_score', 0)
        )
    
    # Add edges
    for _, row in edges_df.iterrows():
        if row['source'] in G.nodes() and row['target'] in G.nodes():
            G.add_edge(
                row['source'],
                row['target'],
                edge_type=row.get('layer', 'unknown'),
                original_type=row.get('type', '')
            )
    
    if viz_mode == "Interactive (PyVis)":
        try:
            from pyvis.network import Network
            import tempfile
            import os
            import streamlit.components.v1 as components
            
            # Create PyVis network
            nt = Network(
                height="550px",
                width="100%",
                bgcolor="#1a1a2e",
                font_color="white",
                directed=True,
                cdn_resources='remote'
            )
            
            nt.set_options("""
            {
                "nodes": {"borderWidth": 2, "font": {"size": 10}},
                "edges": {"smooth": {"type": "curvedCW", "roundness": 0.1}},
                "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 100},
                    "barnesHut": {"gravitationalConstant": -2000, "springLength": 100}
                }
            }
            """)
            
            # Add nodes
            for node in G.nodes():
                attrs = G.nodes[node]
                score = attrs.get('trust_score', 0)
                color = '#16a34a' if score and score > 10 else '#dc2626'
                label = attrs.get('label', str(node)[:8])
                
                nt.add_node(
                    str(node),
                    label=label,
                    title=f"ID: {node}\nScore: {score}",
                    color=color,
                    size=15
                )
            
            # Add edges with colors by type
            edge_colors = {
                'follow': '#3b82f6',
                'interact': '#06b6d4',
                'co_owner': '#dc2626',
                'similarity': '#7c3aed'
            }
            
            for u, v, data in G.edges(data=True):
                etype = data.get('edge_type', 'follow')
                color = edge_colors.get(etype, '#6b7280')
                nt.add_edge(str(u), str(v), color=color, title=data.get('original_type', ''))
            
            # Save and display
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                nt.save_graph(f.name)
                with open(f.name, 'r', encoding='utf-8') as html_file:
                    html_content = html_file.read()
                os.unlink(f.name)
            
            components.html(html_content, height=570, scrolling=False)
            
            # Add legend
            st.markdown(create_legend_html(), unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"PyVis rendering failed: {e}. Falling back to static.")
            viz_mode = "Static (Matplotlib)"
    
    if viz_mode == "Static (Matplotlib)":
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('white')
        
        # Layout
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        # Draw edges by type
        edge_colors_map = {
            'follow': '#3b82f6',
            'interact': '#06b6d4',
            'co_owner': '#dc2626',
            'similarity': '#7c3aed'
        }
        
        for etype, color in edge_colors_map.items():
            edge_list = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == etype]
            if edge_list:
                nx.draw_networkx_edges(
                    G, pos, ax=ax, edgelist=edge_list,
                    edge_color=color, alpha=0.6, width=1,
                    arrows=True, arrowsize=8
                )
        
        # Draw nodes
        node_colors = ['#16a34a' if G.nodes[n].get('trust_score', 0) > 10 else '#dc2626' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=100, alpha=0.8)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def main():
    """Main application entry point."""
    
    # Title
    st.markdown('<h1 class="page-title">Model Laboratory</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Data ingestion, clustering, labeling, and model training workbench</p>',
        unsafe_allow_html=True
    )
    
    # Initialize session state
    if 'lab_data' not in st.session_state:
        st.session_state['lab_data'] = None
    if 'clustering_result' not in st.session_state:
        st.session_state['clustering_result'] = None
    if 'labeling_result' not in st.session_state:
        st.session_state['labeling_result'] = None
    if 'training_result' not in st.session_state:
        st.session_state['training_result'] = None
    
    # Auto-detect data source
    has_exploration_data = 'exploration_data' in st.session_state and st.session_state['exploration_data'] is not None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Data Source</p>', unsafe_allow_html=True)
        
        if has_exploration_data:
            st.caption("Data from Exploration page detected")
        else:
            st.caption("No exploration data - upload files")
        
        st.divider()
        
        # Status indicators
        st.markdown('<p class="sidebar-title">Pipeline Status</p>', unsafe_allow_html=True)
        
        data_status = "Ready" if st.session_state['lab_data'] is not None else "Not loaded"
        cluster_status = "Ready" if st.session_state['clustering_result'] is not None else "Not run"
        label_status = "Ready" if st.session_state['labeling_result'] is not None else "Not run"
        train_status = "Ready" if st.session_state['training_result'] is not None else "Not run"
        
        st.caption(f"1. Data: {data_status}")
        st.caption(f"2. Clustering: {cluster_status}")
        st.caption(f"3. Labeling: {label_status}")
        st.caption(f"4. Training: {train_status}")
    
    # Main content with tabs
    tab_data, tab_cluster, tab_label, tab_train = st.tabs([
        "1. Data Ingestion",
        "2. Clustering",
        "3. Labeling",
        "4. Training"
    ])
    
    # ========== TAB 1: DATA INGESTION ==========
    with tab_data:
        st.markdown('<p class="section-header">Data Ingestion & Preprocessing</p>', unsafe_allow_html=True)
        
        if has_exploration_data:
            # Get data from exploration page
            exp_data = st.session_state['exploration_data']
            
            # Use lab_data if already processed, otherwise use exploration data
            if st.session_state['lab_data'] is not None:
                nodes_df = st.session_state['lab_data']['nodes_df']
                edges_df = st.session_state['lab_data']['edges_df']
                is_processed = True
            else:
                nodes_df = exp_data['nodes_df']
                edges_df = exp_data['edges_df']
                is_processed = False
            
            # Calculate isolated nodes
            isolated_count = count_isolated_nodes(nodes_df, edges_df)
            
            # Visualization mode selector
            viz_mode = st.radio(
                "Graph Visualization Mode",
                options=["Static (Matplotlib)", "Interactive (PyVis)"],
                index=0,
                horizontal=True
            )
            
            st.divider()
            
            # Graph Visualization
            st.markdown("**Network Graph**")
            render_data_graph(nodes_df, edges_df, viz_mode)
            
            st.divider()
            
            # Data Tables
            st.markdown("**Data Tables**")
            tab_nodes, tab_edges = st.tabs(["Nodes", "Edges"])
            
            with tab_nodes:
                st.dataframe(nodes_df, use_container_width=True, height=300)
            
            with tab_edges:
                st.dataframe(edges_df, use_container_width=True, height=300)
            
            st.divider()
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                render_scorecard("Total Nodes", f"{len(nodes_df):,}",
                                 delta="Nodes with edges" if isolated_count > 0 else "" )
            with col2:
                render_scorecard("Isolated Nodes", f"{isolated_count:,}", 
                                 delta="Nodes without edges" if isolated_count > 0 else "")
            with col3:
                render_scorecard("Total Edges", f"{len(edges_df):,}",
                                 delta="Edges between nodes" if isolated_count > 0 else "")
            
            st.divider()
            
            # Process Data button
            if not is_processed and isolated_count > 0:
                if st.button("Process Data", type="primary", use_container_width=True):
                    # Remove isolated nodes
                    processed_nodes, processed_edges, removed = remove_isolated_nodes(
                        nodes_df.copy(), edges_df.copy()
                    )
                    
                    st.session_state['lab_data'] = {
                        'nodes_df': processed_nodes,
                        'edges_df': processed_edges,
                        'removed_nodes': removed
                    }
                    
                    st.success(f"Processed! Removed {removed} isolated nodes. Remaining: {len(processed_nodes)} nodes.")
                    st.rerun()
                
                st.caption("Remove isolated nodes to prepare for training")
            
            elif is_processed:
                st.success(f"Data is ready for training ({len(nodes_df):,} nodes, {len(edges_df):,} edges)")
                
                # Continue to Clustering button
                st.divider()
                if st.button("Continue to Clustering", type="primary", use_container_width=True, key="continue_clustering_1"):
                    st.session_state['active_tab'] = 1
                    st.rerun()
            
            elif isolated_count == 0:
                # No isolated nodes, auto-load to lab_data
                if st.session_state['lab_data'] is None:
                    st.session_state['lab_data'] = {
                        'nodes_df': nodes_df.copy(),
                        'edges_df': edges_df.copy(),
                        'removed_nodes': 0
                    }
                    st.rerun()
                st.success("No isolated nodes found. Data is ready for training.")
                
                # Continue to Clustering button
                st.divider()
                if st.button("Continue to Clustering", type="primary", use_container_width=True, key="continue_clustering_2"):
                    st.session_state['active_tab'] = 1
                    st.rerun()
        
        else:
            # Show file upload interface
            st.info("No data from exploration page. Please upload CSV files.")
            
            st.markdown("**Upload CSV Files**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nodes_file = st.file_uploader(
                    "Nodes CSV",
                    type=['csv'],
                    help="Required columns: profile_id. Optional: handle, trust_score, created_on"
                )
            
            with col2:
                edges_file = st.file_uploader(
                    "Edges CSV",
                    type=['csv'],
                    help="Required columns: source, target. Optional: type"
                )
            
            if nodes_file and edges_file:
                nodes_df = pd.read_csv(nodes_file)
                edges_df = pd.read_csv(edges_file)
                
                # Validate
                errors, warnings = validate_dataframes(nodes_df, edges_df)
                
                for e in errors:
                    st.error(e)
                for w in warnings:
                    st.warning(w)
                
                if not errors:
                    # Calculate isolated nodes
                    isolated_count = count_isolated_nodes(nodes_df, edges_df)
                    
                    # Visualization mode selector
                    viz_mode = st.radio(
                        "Graph Visualization Mode",
                        options=["Static (Matplotlib)", "Interactive (PyVis)"],
                        index=0,
                        horizontal=True,
                        key="upload_viz_mode"
                    )
                    
                    st.divider()
                    
                    # Graph Visualization
                    st.markdown("**Network Graph**")
                    render_data_graph(nodes_df, edges_df, viz_mode)
                    
                    st.divider()
                    
                    # Data Tables
                    st.markdown("**Data Tables**")
                    tab_nodes, tab_edges = st.tabs(["Nodes", "Edges"])
                    
                    with tab_nodes:
                        st.dataframe(nodes_df, use_container_width=True, height=300)
                    
                    with tab_edges:
                        st.dataframe(edges_df, use_container_width=True, height=300)
                    
                    st.divider()
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        render_scorecard("Total Nodes", f"{len(nodes_df):,}",
                                        delta="Nodes with edges" if isolated_count > 0 else ""     )
                    with col2:
                        render_scorecard("Isolated Nodes", f"{isolated_count:,}",
                                        delta="Nodes without edges" if isolated_count > 0 else "")
                    with col3:
                        render_scorecard("Total Edges", f"{len(edges_df):,}",
                                         delta="Edges between nodes" if isolated_count > 0 else "")
                    
                    st.divider()
                    
                    # Process Data button
                    if st.button("Process Data", type="primary", use_container_width=True, key="upload_process"):
                        # Remove isolated nodes
                        processed_nodes, processed_edges, removed = remove_isolated_nodes(
                            nodes_df.copy(), edges_df.copy()
                        )
                        
                        st.session_state['lab_data'] = {
                            'nodes_df': processed_nodes,
                            'edges_df': processed_edges,
                            'removed_nodes': removed
                        }
                        
                        if removed > 0:
                            st.success(f"Processed! Removed {removed} isolated nodes. Remaining: {len(processed_nodes)} nodes.")
                        else:
                            st.success("No isolated nodes found. Data is ready for training.")
                        st.rerun()
                    
                    st.caption("Remove isolated nodes to prepare for training")
    
    # ========== TAB 2: CLUSTERING ==========
    with tab_cluster:
        st.markdown('<p class="section-header">Unsupervised Clustering</p>', unsafe_allow_html=True)
        
        if st.session_state['lab_data'] is None:
            st.info("Please load data in the Data Ingestion tab first.")
        else:
            data = st.session_state['lab_data']
            n_samples = len(data['nodes_df'])
            
            # Display data stats at top
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                render_scorecard("Total Nodes", f"{n_samples:,}")
            with col_stat2:
                isolated = count_isolated_nodes(data['nodes_df'], data['edges_df'])
                render_scorecard("Isolated Nodes", f"{isolated:,}")
            with col_stat3:
                render_scorecard("Total Edges", f"{len(data['edges_df']):,}")
            
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Configuration**")
                
                auto_k = st.checkbox("Auto-select K", value=True, help="Find optimal K using Silhouette Score")
                
                # Max K must be <= n_samples - 1 for Silhouette Score
                max_allowed_k = n_samples - 1
                
                if auto_k:
                    k_min = st.number_input("Min K", min_value=2, max_value=min(20, max_allowed_k), value=2)
                    k_max = st.number_input("Max K", min_value=3, max_value=min(30, max_allowed_k), value=min(15, max_allowed_k))
                    
                    if k_max >= n_samples:
                        st.warning(f"Max K adjusted to {max_allowed_k} (must be < n_samples={n_samples})")
                        k_max = max_allowed_k
                else:
                    n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=min(50, max_allowed_k), value=min(10, max_allowed_k))
                
                run_clustering = st.button("Run Clustering", type="primary", use_container_width=True)
            
            with col2:
                if run_clustering:
                    try:
                        from utils.clustering_engine import ClusteringEngine
                        
                        with st.spinner("Running clustering pipeline..."):
                            engine = ClusteringEngine()
                            
                            if auto_k:
                                # Ensure k_max doesn't exceed limit
                                safe_k_max = min(k_max, max_allowed_k)
                                result, optimal_k, pyg_data = engine.process_and_cluster(
                                    data['nodes_df'],
                                    data['edges_df'],
                                    n_clusters=None,
                                    k_range=(k_min, safe_k_max),
                                    random_state=42
                                )
                            else:
                                result, optimal_k, pyg_data = engine.process_and_cluster(
                                    data['nodes_df'],
                                    data['edges_df'],
                                    n_clusters=n_clusters,
                                    random_state=42
                                )
                            
                            st.session_state['clustering_result'] = {
                                'result': result,
                                'optimal_k': optimal_k,
                                'pyg_data': pyg_data
                            }
                            
                            st.success(f"Clustering complete! Found {result.k} clusters.")
                    
                    except Exception as e:
                        st.error(f"Clustering failed: {str(e)}")
                
                # Display results
                if st.session_state['clustering_result'] is not None:
                    cr = st.session_state['clustering_result']
                    result = cr['result']
                    optimal_k = cr['optimal_k']
                    
                    # Metrics
                    st.markdown("**Results**")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        render_scorecard("Clusters", str(result.k))
                    with m2:
                        render_scorecard("Silhouette", f"{result.silhouette:.4f}")
                    with m3:
                        render_scorecard("Davies-Bouldin", f"{result.davies_bouldin:.4f}")
                    
                    # Optimal K plot
                    if optimal_k:
                        st.markdown("**Optimal K Search**")
                        
                        fig = make_subplots(rows=1, cols=2, subplot_titles=('Silhouette Score', 'Davies-Bouldin Index'))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=optimal_k.k_range, 
                                y=optimal_k.silhouette_scores,
                                mode='lines+markers',
                                name='Silhouette',
                                line=dict(color='#16a34a')
                            ),
                            row=1, col=1
                        )
                        
                        # Mark best K
                        best_idx = optimal_k.k_range.index(optimal_k.best_k)
                        fig.add_trace(
                            go.Scatter(
                                x=[optimal_k.best_k],
                                y=[optimal_k.silhouette_scores[best_idx]],
                                mode='markers',
                                marker=dict(size=15, color='red', symbol='star'),
                                name=f'Best K={optimal_k.best_k}'
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=optimal_k.k_range, 
                                y=optimal_k.davies_bouldin_scores,
                                mode='lines+markers',
                                name='Davies-Bouldin',
                                line=dict(color='#dc2626')
                            ),
                            row=1, col=2
                        )
                        
                        fig.update_layout(
                            height=350,
                            showlegend=False,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster distribution
                    st.markdown("**Cluster Distribution**")
                    
                    cluster_counts = pd.Series(result.labels).value_counts().sort_index()
                    fig_dist = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        labels={'x': 'Cluster ID', 'y': 'Node Count'},
                        template='plotly_dark'
                    )
                    fig_dist.update_layout(height=300)
                    fig_dist.update_xaxes(dtick=1)  # Step of 1 for x-axis
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Clustering results table
                    st.markdown("**Clustering Results Table**")
                    
                    # Create results dataframe
                    results_df = data['nodes_df'].copy()
                    results_df['cluster_id'] = result.labels
                    
                    # Display with cluster column
                    display_cols = ['profile_id', 'cluster_id']
                    if 'handle' in results_df.columns:
                        display_cols.insert(1, 'handle')
                    if 'trust_score' in results_df.columns:
                        display_cols.append('trust_score')
                    
                    st.dataframe(
                        results_df[display_cols].sort_values('cluster_id'),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Continue to Labeling button
                    st.divider()
                    if st.button("Continue to Labeling", type="primary", use_container_width=True, key="continue_labeling"):
                        st.session_state['active_tab'] = 2
                        st.rerun()
    
    # ========== TAB 3: LABELING ==========
    with tab_label:
        st.markdown('<p class="section-header">Semi-Supervised Labeling</p>', unsafe_allow_html=True)
        
        if st.session_state['clustering_result'] is None:
            st.info("Please run clustering first.")
        else:
            cr = st.session_state['clustering_result']
            data = st.session_state['lab_data']
            result = cr['result']
            
            # Display clustering stats at top
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                render_scorecard("Clusters", str(result.k))
            with col_stat2:
                render_scorecard("Silhouette", f"{result.silhouette:.4f}")
            with col_stat3:
                render_scorecard("Davies-Bouldin", f"{result.davies_bouldin:.4f}")
            
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Threshold Configuration**")
                
                st.caption("Edge Ratio Thresholds")
                
                pct_co_owner = st.slider(
                    "Co-Owner Edge %", 
                    min_value=0.0, max_value=0.5, value=0.05, step=0.01,
                    help="Priority 1: Clusters with co-owner edge ratio above this are SYBIL"
                )
                
                pct_fuzzy = st.slider(
                    "Fuzzy Handle Edge %", 
                    min_value=0.0, max_value=1.0, value=0.50, step=0.05,
                    help="Priority 2: Fuzzy handle edge ratio threshold"
                )
                
                pct_similarity = st.slider(
                    "Similarity Edge %", 
                    min_value=0.0, max_value=1.0, value=0.60, step=0.05,
                    help="Priority 3: Similarity edge ratio for bot farm detection"
                )
                
                pct_social = st.slider(
                    "Social Activity %",
                    min_value=0.0, max_value=1.0, value=0.20, step=0.05,
                    help="Priority 3: Clusters below this social activity are suspicious"
                )
                
                st.divider()
                st.caption("Trust Score Thresholds")
                
                co_owner_avg_trust = st.slider(
                    "Co-Owner Trust Threshold",
                    min_value=0, max_value=100, value=25,
                    help="Priority 2: Max trust score for name pattern abuse"
                )
                
                industrial_avg_trust = st.slider(
                    "Industrial Trust Threshold",
                    min_value=0, max_value=100, value=20,
                    help="Priority 3: Max trust score for bot farm detection"
                )
                
                st.divider()
                st.caption("Creation Time Threshold")
                
                std_creation_hours = st.slider(
                    "Std Creation Hours",
                    min_value=0.0, max_value=24.0, value=2.0, step=0.5,
                    help="Priority 3: Accounts created within this std hours are batch-created"
                )
                
                run_labeling = st.button("Apply Rules", type="primary", use_container_width=True)
            
            with col2:
                if run_labeling:
                    try:
                        from utils.labeling_engine import LabelingEngine, LabelingThresholds, create_labeling_summary
                        
                        with st.spinner("Applying labeling rules..."):
                            thresholds = LabelingThresholds(
                                pct_co_owner=pct_co_owner,
                                pct_fuzzy_handle=pct_fuzzy,
                                pct_similarity=pct_similarity,
                                std_creation_hours=std_creation_hours,
                                pct_social=pct_social,
                                co_owner_avg_trust=co_owner_avg_trust,
                                industrial_avg_trust=industrial_avg_trust
                            )
                            
                            engine = LabelingEngine(thresholds)
                            
                            # Profile clusters
                            profiles = engine.profile_clusters(
                                data['nodes_df'],
                                data['edges_df'],
                                cr['result'].labels
                            )
                            
                            # Apply rules
                            results, summary_df = engine.label_clusters(profiles)
                            
                            # Generate node labels
                            node_labels = engine.generate_node_labels(
                                data['nodes_df'],
                                cr['result'].labels,
                                results
                            )
                            
                            summary = create_labeling_summary(results)
                            
                            st.session_state['labeling_result'] = {
                                'results': results,
                                'summary_df': summary_df,
                                'node_labels': node_labels,
                                'summary': summary
                            }
                            
                            st.success("Labeling complete!")
                    
                    except Exception as e:
                        st.error(f"Labeling failed: {str(e)}")
                
                # Display results
                if st.session_state['labeling_result'] is not None:
                    lr = st.session_state['labeling_result']
                    summary = lr['summary']
                    
                    # Metrics
                    st.markdown("**Results**")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        render_scorecard("Sybil Clusters", str(summary['sybil_clusters']))
                    with m2:
                        render_scorecard("Non-Sybil Clusters", str(summary['nonsybil_clusters']))
                    with m3:
                        render_scorecard("Sybil Nodes", f"{summary['sybil_nodes']:,}")
                    with m4:
                        render_scorecard("Sybil Ratio", f"{summary['sybil_ratio']:.1%}")
                    
                    st.markdown("**Label Distribution**")
                    
                    # Pie chart
                    fig_pie = px.pie(
                        values=[summary['sybil_nodes'], summary['nonsybil_nodes']],
                        names=['SYBIL', 'NON-SYBIL'],
                        color_discrete_map={'SYBIL': '#dc2626', 'NON-SYBIL': '#16a34a'},
                        template='plotly_dark'
                    )
                    fig_pie.update_layout(height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Cluster summary table
                    st.markdown("**Cluster Summary**")
                    st.dataframe(lr['summary_df'], use_container_width=True, height=300)
                    
                    # Continue to Training button
                    st.divider()
                    if st.button("Continue to Training", type="primary", use_container_width=True, key="continue_training"):
                        st.session_state['active_tab'] = 3
                        st.rerun()
    
    # ========== TAB 4: TRAINING ==========
    with tab_train:
        st.markdown('<p class="section-header">Supervised GAT Training</p>', unsafe_allow_html=True)
        
        if st.session_state['labeling_result'] is None:
            st.info("Please complete labeling first.")
        else:
            lr = st.session_state['labeling_result']
            cr = st.session_state['clustering_result']
            data = st.session_state['lab_data']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Hyperparameters**")
                
                hidden_channels = st.selectbox(
                    "Hidden Channels",
                    options=[16, 32, 64, 128],
                    index=1
                )
                
                heads = st.selectbox(
                    "Attention Heads",
                    options=[2, 4, 8],
                    index=1
                )
                
                dropout = st.slider(
                    "Dropout",
                    min_value=0.0, max_value=0.6, value=0.3, step=0.1
                )
                
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.001, 0.005, 0.01, 0.05],
                    value=0.005
                )
                
                epochs = st.number_input(
                    "Max Epochs",
                    min_value=50, max_value=500, value=300
                )
                
                patience = st.number_input(
                    "Early Stopping Patience",
                    min_value=10, max_value=100, value=40
                )
                
                run_training = st.button("Start Training", type="primary", use_container_width=True)
            
            with col2:
                # Training progress area
                progress_container = st.container()
                
                if run_training:
                    try:
                        from utils.trainer import GATrainer, TrainingConfig
                        
                        config = TrainingConfig(
                            hidden_channels=hidden_channels,
                            heads=heads,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            patience=patience
                        )
                        
                        trainer = GATrainer(config)
                        
                        # Prepare data
                        pyg_data = cr['pyg_data']
                        node_labels = lr['node_labels']
                        
                        # Progress tracking
                        progress_bar = progress_container.progress(0)
                        status_text = progress_container.empty()
                        
                        # Chart placeholder
                        chart_placeholder = progress_container.empty()
                        
                        # Training history for live plot
                        live_epochs = []
                        live_loss = []
                        live_f1 = []
                        
                        def training_callback(epoch, metrics):
                            live_epochs.append(epoch)
                            live_loss.append(metrics['loss'])
                            live_f1.append(metrics['val_f1'])
                            
                            # Update progress
                            progress = min(epoch / epochs, 1.0)
                            progress_bar.progress(progress)
                            
                            status_text.markdown(
                                f"**Epoch {epoch}/{epochs}** | "
                                f"Loss: {metrics['loss']:.4f} | "
                                f"Val F1: {metrics['val_f1']:.4f} | "
                                f"Best F1: {metrics['best_val_f1']:.4f}"
                            )
                            
                            # Update chart every 5 epochs
                            if epoch % 5 == 0 or epoch == 1:
                                fig = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=('Training Loss', 'Validation F1')
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=live_epochs, y=live_loss,
                                        mode='lines',
                                        name='Loss',
                                        line=dict(color='#3b82f6')
                                    ),
                                    row=1, col=1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=live_epochs, y=live_f1,
                                        mode='lines',
                                        name='F1',
                                        line=dict(color='#16a34a')
                                    ),
                                    row=1, col=2
                                )
                                
                                fig.update_layout(
                                    height=300,
                                    showlegend=False,
                                    template='plotly_dark'
                                )
                                
                                chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Run training
                        with st.spinner("Training GAT model..."):
                            history = trainer.train(pyg_data, node_labels, callback=training_callback)
                            
                            # Evaluate
                            eval_result = trainer.evaluate_test(pyg_data)
                            
                            st.session_state['training_result'] = {
                                'history': history,
                                'eval_result': eval_result,
                                'trainer': trainer
                            }
                        
                        progress_bar.progress(1.0)
                        st.success(f"Training complete! Best F1: {history.best_val_f1:.4f} at epoch {history.best_epoch}")
                    
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # Display final results
                if st.session_state['training_result'] is not None:
                    tr = st.session_state['training_result']
                    history = tr['history']
                    eval_result = tr['eval_result']
                    
                    st.markdown("**Final Results**")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        render_scorecard("Test Accuracy", f"{eval_result.accuracy:.4f}")
                    with m2:
                        render_scorecard("Test F1 (Macro)", f"{eval_result.f1_macro:.4f}")
                    with m3:
                        render_scorecard("Best Epoch", str(history.best_epoch))
                    
                    # Confusion Matrix
                    st.markdown("**Confusion Matrix**")
                    
                    cm = eval_result.confusion_matrix
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Non-Sybil', 'Sybil'],
                        y=['Non-Sybil', 'Sybil'],
                        text_auto=True,
                        color_continuous_scale='Blues',
                        template='plotly_dark'
                    )
                    fig_cm.update_layout(height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Classification report
                    with st.expander("Classification Report"):
                        st.code(eval_result.classification_report)
                    
                    # Export section
                    st.markdown("**Export Trained Assets**")
                    
                    st.caption("Save all required files for inference in the Sybil Detector")
                    
                    # Model filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_name = st.text_input(
                        "Model filename",
                        value=f"gat_model_{timestamp}.pt"
                    )
                    
                    col_save1, col_save2, col_save3 = st.columns(3)
                    
                    with col_save1:
                        if st.button("Save Model", use_container_width=True):
                            save_path = f"assets/{model_name}"
                            tr['trainer'].save_model(save_path)
                            st.success(f"Model saved to {save_path}")
                    
                    with col_save2:
                        if st.button("Save Sybil Data", use_container_width=True):
                            import torch
                            pyg_data = cr['pyg_data']
                            # Add labels to data object
                            pyg_data.y = torch.tensor(lr['node_labels'], dtype=torch.long)
                            
                            data_path = f"assets/processed_sybil_data_{timestamp}.pt"
                            torch.save(pyg_data, data_path)
                            st.success(f"Sybil data saved to {data_path}")
                    
                    with col_save3:
                        if st.button("Save Scaler", use_container_width=True):
                            import joblib
                            from utils.clustering_engine import ClusteringEngine
                            
                            # Get the scaler from clustering engine's feature engineer
                            engine = ClusteringEngine()
                            # Re-fit scaler on current data to ensure consistency
                            _ = engine.feature_engineer.process_nodes(data['nodes_df'])
                            scaler = engine.feature_engineer.scaler
                            
                            scaler_path = f"assets/std_scaler_{timestamp}.bin"
                            joblib.dump(scaler, scaler_path)
                            st.success(f"Scaler saved to {scaler_path}")
                    
                    st.divider()
                    st.caption("Files required for Sybil Detector:")
                    st.markdown("""
                    - **Model**: GAT model weights (`.pt`)
                    - **Sybil Data**: Reference graph with features and labels (`.pt`)
                    - **Scaler**: StandardScaler for feature normalization (`.bin`)
                    """)


if __name__ == "__main__":
    main()

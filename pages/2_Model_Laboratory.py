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

from utils.ui import (
    setup_page,
    page_header,
    section_header,
    sidebar_header,
    metric_card,
    apply_plotly_theme,
    Colors
)
from utils.visualizer import create_legend_html


# Page setup
setup_page("Model Laboratory")


def validate_dataframes(
    nodes_df: pd.DataFrame, 
    edges_df: pd.DataFrame
) -> tuple:
    """Validate uploaded dataframes."""
    errors = []
    warnings = []
    
    if 'profile_id' not in nodes_df.columns:
        errors.append("nodes.csv missing 'profile_id' column")
    
    # Check for either source/target OR source_id/target_id columns
    has_source_target = 'source' in edges_df.columns and 'target' in edges_df.columns
    has_source_id_target_id = 'source_id' in edges_df.columns and 'target_id' in edges_df.columns
    
    if not (has_source_target or has_source_id_target_id):
        errors.append("edges.csv missing 'source'/'target' or 'source_id'/'target_id' columns")
    
    useful_cols = ['handle', 'trust_score', 'created_on', 'owned_by']
    missing_useful = [c for c in useful_cols if c not in nodes_df.columns]
    if missing_useful:
        warnings.append(f"Optional columns missing: {', '.join(missing_useful)}")
    
    if 'type' not in edges_df.columns and 'layer' not in edges_df.columns:
        warnings.append("edges.csv missing 'type' or 'layer' column - will use default weights")
    
    return errors, warnings


def remove_isolated_nodes(
    nodes_df: pd.DataFrame, 
    edges_df: pd.DataFrame
) -> tuple:
    """Remove nodes with degree 0 (no connections)."""
    connected_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
    
    original_count = len(nodes_df)
    nodes_df = nodes_df[nodes_df['profile_id'].isin(connected_nodes)].copy()
    removed_count = original_count - len(nodes_df)
    
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
    
    G = nx.DiGraph()
    
    for _, row in nodes_df.iterrows():
        handle = row.get('handle', str(row['profile_id'])[:8])
        G.add_node(
            row['profile_id'],
            label=handle[:12] if len(str(handle)) > 12 else handle,
            trust_score=row.get('trust_score', 0)
        )
    
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
            
            nt = Network(
                height="450px",
                width="100%",
                bgcolor="#FFFFFF",
                font_color="#111827",
                directed=True,
                cdn_resources='remote'
            )
            
            nt.set_options("""
            {
                "nodes": {"borderWidth": 2, "font": {"size": 10, "color": "#111827"}},
                "edges": {"smooth": {"type": "curvedCW", "roundness": 0.1}},
                "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 100},
                    "barnesHut": {"gravitationalConstant": -2000, "springLength": 100}
                }
            }
            """)
            
            for node in G.nodes():
                attrs = G.nodes[node]
                score = attrs.get('trust_score', 0)
                color = Colors.SAFE if score and score > 10 else Colors.DANGER
                label = attrs.get('label', str(node)[:8])
                
                nt.add_node(
                    str(node),
                    label=label,
                    title=f"ID: {node}\nScore: {score}",
                    color=color,
                    size=15
                )
            
            edge_colors = {
                'follow': Colors.BLUE,
                'interact': Colors.CYAN,
                'co_owner': Colors.DANGER,
                'similarity': Colors.PURPLE
            }
            
            for u, v, data in G.edges(data=True):
                etype = data.get('edge_type', 'follow')
                color = edge_colors.get(etype, Colors.NEUTRAL)
                nt.add_edge(str(u), str(v), color=color, title=data.get('original_type', ''))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                nt.save_graph(f.name)
                with open(f.name, 'r', encoding='utf-8') as html_file:
                    html_content = html_file.read()
                os.unlink(f.name)
            
            components.html(html_content, height=470, scrolling=False)
            st.markdown(create_legend_html(), unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"PyVis rendering failed: {e}. Falling back to static.")
            viz_mode = "Static (Matplotlib)"
    
    if viz_mode == "Static (Matplotlib)":
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('white')
        
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        edge_colors_map = {
            'follow': Colors.BLUE,
            'interact': Colors.CYAN,
            'co_owner': Colors.DANGER,
            'similarity': Colors.PURPLE
        }
        
        for etype, color in edge_colors_map.items():
            edge_list = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == etype]
            if edge_list:
                nx.draw_networkx_edges(
                    G, pos, ax=ax, edgelist=edge_list,
                    edge_color=color, alpha=0.6, width=1,
                    arrows=True, arrowsize=8
                )
        
        node_colors = [Colors.SAFE if G.nodes[n].get('trust_score', 0) > 10 else Colors.DANGER for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=100, alpha=0.8)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def main():
    """Main application entry point."""
    
    # Header
    page_header(
        "Model Laboratory",
        "Data ingestion, clustering, labeling, and model training workbench"
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
    
    has_exploration_data = 'exploration_data' in st.session_state and st.session_state['exploration_data'] is not None
    has_processed_data = st.session_state['lab_data'] is not None
    
    # Determine data source priority: processed lab_data > uploaded files > exploration_data
    data_source_mode = None
    if has_processed_data:
        data_source_mode = "processed"
    elif 'uploaded_files_active' in st.session_state and st.session_state['uploaded_files_active']:
        data_source_mode = "uploaded"
    elif has_exploration_data:
        data_source_mode = "exploration"
    else:
        data_source_mode = "none"
    
    # Sidebar
    with st.sidebar:
        sidebar_header("Data Source")
        
        if data_source_mode == "processed":
            st.caption("Processed data ready")
        elif data_source_mode == "uploaded":
            st.caption("Uploaded files detected")
        elif data_source_mode == "exploration":
            st.caption("Data from Exploration page")
        else:
            st.caption("No data - upload files")
        
        st.divider()
        
        sidebar_header("Pipeline Status")
        
        data_status = "Ready" if st.session_state['lab_data'] is not None else "Not loaded"
        cluster_status = "Ready" if st.session_state['clustering_result'] is not None else "Not run"
        label_status = "Ready" if st.session_state['labeling_result'] is not None else "Not run"
        train_status = "Ready" if st.session_state['training_result'] is not None else "Not run"
        
        st.caption(f"1. Data: {data_status}")
        st.caption(f"2. Clustering: {cluster_status}")
        st.caption(f"3. Labeling: {label_status}")
        st.caption(f"4. Training: {train_status}")
    
    # Main tabs
    tab_data, tab_cluster, tab_label, tab_train = st.tabs([
        "1. Data Ingestion",
        "2. Clustering",
        "3. Labeling",
        "4. Training"
    ])
    
    # ========== TAB 1: DATA INGESTION ==========
    with tab_data:
        section_header("Data Ingestion & Preprocessing")
        
        # === CASE 1: PROCESSED DATA (Already processed, show results) ===
        if data_source_mode == "processed":
            data = st.session_state['lab_data']
            nodes_df = data['nodes_df']
            edges_df = data['edges_df']
            removed_count = data.get('removed_nodes', 0)
            
            st.success(f"Data is ready for training ({len(nodes_df):,} nodes, {len(edges_df):,} edges)")
            if removed_count > 0:
                st.info(f"Previously removed {removed_count} isolated nodes")
            
            viz_mode = st.radio(
                "Graph Visualization Mode",
                options=["Static (Matplotlib)", "Interactive (PyVis)"],
                index=0,
                horizontal=True,
                key="processed_viz_mode"
            )
            
            st.divider()
            
            # Graph Visualization
            with st.container(border=True):
                st.markdown("### Network Graph")
                render_data_graph(nodes_df, edges_df, viz_mode)
            
            st.divider()
            
            # Data Tables
            with st.container(border=True):
                st.markdown("### Data Tables")
                tab_nodes, tab_edges = st.tabs(["Nodes", "Edges"])
                
                with tab_nodes:
                    st.dataframe(nodes_df, use_container_width=True, height=250, hide_index=True)
                
                with tab_edges:
                    st.dataframe(edges_df, use_container_width=True, height=250, hide_index=True)
            
            st.divider()
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                metric_card("Total Nodes", f"{len(nodes_df):,}", compact=True)
            with col2:
                metric_card("Isolated Nodes", "0", compact=True)
            with col3:
                metric_card("Total Edges", f"{len(edges_df):,}", compact=True)
            
            st.divider()
            
            # Reset options
            col_reset1, col_reset2 = st.columns(2)
            with col_reset1:
                if st.button("Reset to Raw Data", use_container_width=True):
                    st.session_state['lab_data'] = None
                    if 'uploaded_files_active' in st.session_state:
                        del st.session_state['uploaded_files_active']
                    st.rerun()
            with col_reset2:
                if st.button("🗑️ Clear All Data", use_container_width=True):
                    for key in ['lab_data', 'exploration_data', 'uploaded_files_active']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        
        # === CASE 2: EXPLORATION DATA (From Data Exploration page) ===
        elif data_source_mode == "exploration":
            exp_data = st.session_state['exploration_data']
            nodes_df = exp_data['nodes_df']
            edges_df = exp_data['edges_df']
            isolated_count = count_isolated_nodes(nodes_df, edges_df)
            
            viz_mode = st.radio(
                "Graph Visualization Mode",
                options=["Static (Matplotlib)", "Interactive (PyVis)"],
                index=0,
                horizontal=True,
                key="exploration_viz_mode"
            )
            
            st.divider()
            
            # Graph Visualization
            with st.container(border=True):
                st.markdown("### Network Graph")
                render_data_graph(nodes_df, edges_df, viz_mode)
            
            st.divider()
            
            # Data Tables
            with st.container(border=True):
                st.markdown("### Data Tables")
                tab_nodes, tab_edges = st.tabs(["Nodes", "Edges"])
                
                with tab_nodes:
                    st.dataframe(nodes_df, use_container_width=True, height=250, hide_index=True)
                
                with tab_edges:
                    st.dataframe(edges_df, use_container_width=True, height=250, hide_index=True)
            
            st.divider()
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                metric_card("Total Nodes", f"{len(nodes_df):,}", compact=True)
            with col2:
                metric_card("Isolated Nodes", f"{isolated_count:,}", compact=True)
            with col3:
                metric_card("Total Edges", f"{len(edges_df):,}", compact=True)
            
            st.divider()
            
            # Process button for exploration data
            if isolated_count > 0:
                if st.button("Process Data", type="primary", use_container_width=True, key="exploration_process"):
                    processed_nodes, processed_edges, removed = remove_isolated_nodes(
                        nodes_df.copy(), edges_df.copy()
                    )
                    
                    st.session_state['lab_data'] = {
                        'nodes_df': processed_nodes,
                        'edges_df': processed_edges,
                        'removed_nodes': removed
                    }
                    
                    st.success(f"Processed! Removed {removed} isolated nodes. Remaining: {len(processed_nodes)} nodes.")
                    import time

                    time.sleep(0.5)
                    st.rerun()
                
                st.caption("Remove isolated nodes to prepare for training")
            else:
                if st.button("Use This Data", type="primary", use_container_width=True, key="exploration_use"):
                    st.session_state['lab_data'] = {
                        'nodes_df': nodes_df.copy(),
                        'edges_df': edges_df.copy(),
                        'removed_nodes': 0
                    }
                    st.success("Data is ready for training!")
                    time.sleep(0.5)
                    st.rerun()
                
                st.caption("No isolated nodes found. Data is ready for training.")
        
        # === CASE 3: UPLOAD FILES MODE ===
        else:
            # Check if we're in uploaded files mode
            if data_source_mode == "uploaded":
                st.info("Files uploaded successfully. Please validate and process the data below.")
            else:
                st.info("No data available. Please upload CSV files to begin.")
            
            with st.container(border=True):
                st.markdown("### Upload CSV Files")
                
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
                # Set flag to indicate uploaded files are active
                st.session_state['uploaded_files_active'] = True
                
                nodes_df = pd.read_csv(nodes_file)
                edges_df = pd.read_csv(edges_file)
                
                # Fix Edge Columns - rename source/target to expected format if needed
                if 'source' in edges_df.columns and 'source_id' not in edges_df.columns:
                    edges_df = edges_df.rename(columns={'source': 'source_id'})
                if 'target' in edges_df.columns and 'target_id' not in edges_df.columns:
                    edges_df = edges_df.rename(columns={'target': 'target_id'})
                
                # Fix Node Columns - Remove merge artifacts (columns ending with _x)
                nodes_df.columns = [c.replace('_x', '') if c.endswith('_x') else c for c in nodes_df.columns]
                # Drop _y columns if they exist (merge artifacts)
                nodes_df = nodes_df[[c for c in nodes_df.columns if not c.endswith('_y')]]
                
                # Ensure IDs are strings (consistent with PyG mapping logic)
                if 'profile_id' in nodes_df.columns:
                    nodes_df['profile_id'] = nodes_df['profile_id'].astype(str)
                if 'source_id' in edges_df.columns:
                    edges_df['source_id'] = edges_df['source_id'].astype(str)
                if 'target_id' in edges_df.columns:
                    edges_df['target_id'] = edges_df['target_id'].astype(str)
                
                # Handle legacy column names for validation
                if 'source_id' in edges_df.columns and 'source' not in edges_df.columns:
                    edges_df['source'] = edges_df['source_id']
                if 'target_id' in edges_df.columns and 'target' not in edges_df.columns:
                    edges_df['target'] = edges_df['target_id']
                
                errors, warnings = validate_dataframes(nodes_df, edges_df)
                
                for e in errors:
                    st.error(e)
                for w in warnings:
                    st.warning(w)
                
                if not errors:
                    isolated_count = count_isolated_nodes(nodes_df, edges_df)
                    
                    # ===== DEBUG INFO =====
                    with st.expander("🔍 Debug Info (Click to expand)", expanded=False):
                        st.markdown("**Nodes DataFrame:**")
                        st.write(f"Shape: {nodes_df.shape}")
                        st.write(f"Columns: {list(nodes_df.columns)}")
                        st.write(f"Data types: {dict(nodes_df.dtypes)}")
                        
                        st.markdown("**Edges DataFrame:**")
                        st.write(f"Shape: {edges_df.shape}")
                        st.write(f"Columns: {list(edges_df.columns)}")
                        st.write(f"Data types: {dict(edges_df.dtypes)}")
                        
                        if len(nodes_df) <= 10:
                            st.markdown("**Sample Nodes:**")
                            st.dataframe(nodes_df.head())
                        if len(edges_df) <= 10:
                            st.markdown("**Sample Edges:**")
                            st.dataframe(edges_df.head())
                    
                    viz_mode = st.radio(
                        "Graph Visualization Mode",
                        options=["Static (Matplotlib)", "Interactive (PyVis)"],
                        index=0,
                        horizontal=True,
                        key="upload_viz_mode"
                    )
                    
                    st.divider()
                    
                    with st.container(border=True):
                        st.markdown("### Network Graph")
                        render_data_graph(nodes_df, edges_df, viz_mode)
                    
                    st.divider()
                    
                    with st.container(border=True):
                        st.markdown("### Data Tables")
                        tab_nodes, tab_edges = st.tabs(["Nodes", "Edges"])
                        
                        with tab_nodes:
                            st.dataframe(nodes_df, use_container_width=True, height=250, hide_index=True)
                        
                        with tab_edges:
                            st.dataframe(edges_df, use_container_width=True, height=250, hide_index=True)
                    
                    st.divider()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        metric_card("Total Nodes", f"{len(nodes_df):,}")
                    with col2:
                        metric_card("Isolated Nodes", f"{isolated_count:,}")
                    with col3:
                        metric_card("Total Edges", f"{len(edges_df):,}")
                    
                    st.divider()
                    
                    if st.button("Process Data", type="primary", use_container_width=True, key="upload_process"):
                        with st.spinner("Processing uploaded data..."):
                            try:
                                # ===== FIX 3: ROBUST PROCESSING LOGIC =====
                                import time
                                
                                # Ensure we have the correct column names for processing
                                work_nodes_df = nodes_df.copy()
                                work_edges_df = edges_df.copy()
                                
                                # Ensure edges have the expected source/target columns for remove_isolated_nodes
                                if 'source_id' in work_edges_df.columns and 'source' not in work_edges_df.columns:
                                    work_edges_df['source'] = work_edges_df['source_id']
                                if 'target_id' in work_edges_df.columns and 'target' not in work_edges_df.columns:
                                    work_edges_df['target'] = work_edges_df['target_id']
                                
                                # Process and remove isolated nodes
                                processed_nodes, processed_edges, removed = remove_isolated_nodes(
                                    work_nodes_df, work_edges_df
                                )
                                
                                # ===== FIX 4: EXPLICIT STATE PERSISTENCE =====
                                st.session_state['lab_data'] = {
                                    'nodes_df': processed_nodes,
                                    'edges_df': processed_edges,
                                    'removed_nodes': removed
                                }
                                
                                # Clear uploaded_files_active flag since data is now processed
                                if 'uploaded_files_active' in st.session_state:
                                    del st.session_state['uploaded_files_active']
                                
                                # Success feedback
                                if removed > 0:
                                    st.success(f"Processed! Removed {removed} isolated nodes. Remaining: {len(processed_nodes)} nodes.")
                                else:
                                    st.success("No isolated nodes found. Data is ready for training.")
                                
                                # Force UI update to show processed data mode
                                time.sleep(1.0)  # Give user time to see success message
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error processing upload: {str(e)}")
                                st.exception(e)  # Show full traceback for debugging
                    
                    st.caption("Remove isolated nodes to prepare for training")
    
    # ========== TAB 2: CLUSTERING ==========
    with tab_cluster:
        section_header("Unsupervised Clustering")
        
        if st.session_state['lab_data'] is None:
            st.info("Please load data in the Data Ingestion tab first.")
        else:
            data = st.session_state['lab_data']
            n_samples = len(data['nodes_df'])
            
            # Stats row
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                metric_card("Total Nodes", f"{n_samples:,}", compact=True)
            with col_stat2:
                isolated = count_isolated_nodes(data['nodes_df'], data['edges_df'])
                metric_card("Isolated Nodes", f"{isolated:,}", compact=True)
            with col_stat3:
                metric_card("Total Edges", f"{len(data['edges_df']):,}", compact=True)
            
            st.divider()
            
            # Configuration + Results layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                with st.container(border=True):
                    st.markdown("### Configuration")
                    
                    auto_k = st.checkbox("Auto-select K", value=True, help="Find optimal K using Silhouette Score")
                    
                    max_allowed_k = n_samples - 1
                    
                    if auto_k:
                        k_min = st.number_input("Min K", min_value=2, max_value=min(20, max_allowed_k), value=2)
                        k_max = st.number_input("Max K", min_value=3, max_value=min(30, max_allowed_k), value=min(15, max_allowed_k))
                        
                        if k_max >= n_samples:
                            st.warning(f"Max K adjusted to {max_allowed_k}")
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
                
                if st.session_state['clustering_result'] is not None:
                    cr = st.session_state['clustering_result']
                    result = cr['result']
                    optimal_k = cr['optimal_k']
                    
                    with st.container(border=True):
                        st.markdown("### Results")
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            metric_card("Clusters", str(result.k), compact=True)
                        with m2:
                            metric_card("Silhouette", f"{result.silhouette:.4f}", compact=True)
                        with m3:
                            metric_card("Davies-Bouldin", f"{result.davies_bouldin:.4f}", compact=True)
                    
                    if optimal_k:
                        with st.container(border=True):
                            st.markdown("### Optimal K Search")
                            
                            fig = make_subplots(rows=1, cols=2, subplot_titles=('Silhouette Score', 'Davies-Bouldin Index'))
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=optimal_k.k_range, 
                                    y=optimal_k.silhouette_scores,
                                    mode='lines+markers',
                                    name='Silhouette',
                                    line=dict(color=Colors.SAFE)
                                ),
                                row=1, col=1
                            )
                            
                            best_idx = optimal_k.k_range.index(optimal_k.best_k)
                            fig.add_trace(
                                go.Scatter(
                                    x=[optimal_k.best_k],
                                    y=[optimal_k.silhouette_scores[best_idx]],
                                    mode='markers',
                                    marker=dict(size=15, color=Colors.DANGER, symbol='star'),
                                    name=f'Best K={optimal_k.best_k}'
                                ),
                                row=1, col=1
                            )
                            fig.update_xaxes(dtick=1, row=1, col=1)
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=optimal_k.k_range, 
                                    y=optimal_k.davies_bouldin_scores,
                                    mode='lines+markers',
                                    name='Davies-Bouldin',
                                    line=dict(color=Colors.DANGER)
                                ),
                                row=1, col=2
                            )
                            fig.update_xaxes(dtick=1, row=1, col=2)
                            
                            apply_plotly_theme(fig)
                            fig.update_layout(height=300, showlegend=False)
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with st.container(border=True):
                        st.markdown("### Cluster Distribution")
                        
                        cluster_counts = pd.Series(result.labels).value_counts().sort_index()
                        fig_dist = px.bar(
                            x=cluster_counts.index,
                            y=cluster_counts.values,
                            labels={'x': 'Cluster ID', 'y': 'Node Count'},
                            color_discrete_sequence=[Colors.PRIMARY]
                        )
                        apply_plotly_theme(fig_dist)
                        fig_dist.update_layout(height=250)
                        fig_dist.update_xaxes(dtick=1)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with st.container(border=True):
                        st.markdown("### Clustering Results Table")
                        
                        results_df = data['nodes_df'].copy()
                        results_df['cluster_id'] = result.labels
                        
                        display_cols = ['profile_id', 'cluster_id']
                        if 'handle' in results_df.columns:
                            display_cols.insert(1, 'handle')
                        if 'trust_score' in results_df.columns:
                            display_cols.append('trust_score')
                        
                        st.dataframe(
                            results_df[display_cols].sort_values('cluster_id'),
                            use_container_width=True,
                            height=250,
                            hide_index=True
                        )
    
    # ========== TAB 3: LABELING ==========
    with tab_label:
        section_header("Semi-Supervised Labeling")
        
        if st.session_state['clustering_result'] is None:
            st.info("Please run clustering first.")
        else:
            cr = st.session_state['clustering_result']
            data = st.session_state['lab_data']
            result = cr['result']
            
            # Stats row
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                metric_card("Clusters", str(result.k), compact=True)
            with col_stat2:
                metric_card("Silhouette", f"{result.silhouette:.4f}", compact=True)
            with col_stat3:
                metric_card("Davies-Bouldin", f"{result.davies_bouldin:.4f}", compact=True)
            
            st.divider()
            
            # Configuration + Results layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                with st.container(border=True):
                    st.markdown("### Threshold Configuration")
                    
                    st.caption("Edge Ratio Thresholds")
                    
                    pct_co_owner = st.slider(
                        "Co-Owner Edge %", 
                        min_value=0.0, max_value=0.5, value=0.05, step=0.01,
                        help="Clusters with co-owner edge ratio above this are SYBIL"
                    )
                    
                    pct_fuzzy = st.slider(
                        "Fuzzy Handle Edge %", 
                        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
                        help="Fuzzy handle edge ratio threshold"
                    )
                    
                    pct_similarity = st.slider(
                        "Similarity Edge %", 
                        min_value=0.0, max_value=1.0, value=0.60, step=0.05,
                        help="Similarity edge ratio for bot farm detection"
                    )
                    
                    pct_social = st.slider(
                        "Social Activity %",
                        min_value=0.0, max_value=1.0, value=0.20, step=0.05,
                        help="Clusters below this social activity are suspicious"
                    )
                    
                    st.divider()
                    st.caption("Trust Score Thresholds")
                    
                    co_owner_avg_trust = st.slider(
                        "Co-Owner Trust Threshold",
                        min_value=0, max_value=100, value=25,
                        help="Max trust score for name pattern abuse"
                    )
                    
                    industrial_avg_trust = st.slider(
                        "Industrial Trust Threshold",
                        min_value=0, max_value=100, value=20,
                        help="Max trust score for bot farm detection"
                    )
                    
                    st.divider()
                    st.caption("Creation Time Threshold")
                    
                    std_creation_hours = st.slider(
                        "Std Creation Hours",
                        min_value=0.0, max_value=24.0, value=2.0, step=0.5,
                        help="Accounts created within this std hours are batch-created"
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
                            
                            profiles = engine.profile_clusters(
                                data['nodes_df'],
                                data['edges_df'],
                                cr['result'].labels
                            )
                            
                            results, summary_df = engine.label_clusters(profiles)
                            
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
                
                if st.session_state['labeling_result'] is not None:
                    lr = st.session_state['labeling_result']
                    summary = lr['summary']
                    
                    with st.container(border=True):
                        st.markdown("### Results")
                        
                        # Use 2x2 grid for results
                        row1_col1, row1_col2 = st.columns(2)
                        with row1_col1:
                            metric_card("Sybil Clusters", str(summary['sybil_clusters']), status="danger", compact=True)
                        with row1_col2:
                            metric_card("Non-Sybil Clusters", str(summary['nonsybil_clusters']), status="safe", compact=True)
                        
                        row2_col1, row2_col2 = st.columns(2)
                        with row2_col1:
                            metric_card("Sybil Nodes", f"{summary['sybil_nodes']:,}", compact=True)
                        with row2_col2:
                            metric_card("Sybil Ratio", f"{summary['sybil_ratio']:.1%}", compact=True)
                    
                    with st.container(border=True):
                        st.markdown("### Label Distribution")
                        
                        fig_pie = px.pie(
                            values=[summary['sybil_nodes'], summary['nonsybil_nodes']],
                            names=['SYBIL', 'NON-SYBIL'],
                            color_discrete_map={'SYBIL': Colors.DANGER, 'NON-SYBIL': Colors.SAFE}
                        )
                        apply_plotly_theme(fig_pie)
                        fig_pie.update_layout(height=280)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with st.container(border=True):
                        st.markdown("### Cluster Summary")
                        st.dataframe(lr['summary_df'], use_container_width=True, height=250, hide_index=True)
    
    # ========== TAB 4: TRAINING ==========
    with tab_train:
        section_header("Supervised GAT Training")
        
        if st.session_state['labeling_result'] is None:
            st.info("Please complete labeling first.")
        else:
            lr = st.session_state['labeling_result']
            cr = st.session_state['clustering_result']
            data = st.session_state['lab_data']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                with st.container(border=True):
                    st.markdown("### Hyperparameters")
                    
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
                        
                        pyg_data = cr['pyg_data']
                        node_labels = lr['node_labels']
                        
                        progress_bar = progress_container.progress(0)
                        status_text = progress_container.empty()
                        chart_placeholder = progress_container.empty()
                        
                        live_epochs = []
                        live_loss = []
                        live_f1 = []
                        
                        def training_callback(epoch, metrics):
                            live_epochs.append(epoch)
                            live_loss.append(metrics['loss'])
                            live_f1.append(metrics['val_f1'])
                            
                            progress = min(epoch / epochs, 1.0)
                            progress_bar.progress(progress)
                            
                            status_text.markdown(
                                f"**Epoch {epoch}/{epochs}** | "
                                f"Loss: {metrics['loss']:.4f} | "
                                f"Val F1: {metrics['val_f1']:.4f} | "
                                f"Best F1: {metrics['best_val_f1']:.4f}"
                            )
                            
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
                                        line=dict(color=Colors.BLUE)
                                    ),
                                    row=1, col=1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=live_epochs, y=live_f1,
                                        mode='lines',
                                        name='F1',
                                        line=dict(color=Colors.SAFE)
                                    ),
                                    row=1, col=2
                                )
                                
                                apply_plotly_theme(fig)
                                fig.update_layout(height=280, showlegend=False)
                                
                                chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        with st.spinner("Training GAT model..."):
                            history = trainer.train(pyg_data, node_labels, callback=training_callback)
                            
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
                
                if st.session_state['training_result'] is not None:
                    tr = st.session_state['training_result']
                    history = tr['history']
                    eval_result = tr['eval_result']
                    
                    with st.container(border=True):
                        st.markdown("### Final Results")
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            metric_card("Test Accuracy", f"{eval_result.accuracy:.4f}", compact=True)
                        with m2:
                            metric_card("Test F1 (Macro)", f"{eval_result.f1_macro:.4f}", compact=True)
                        with m3:
                            metric_card("Best Epoch", str(history.best_epoch), compact=True)
                    
                    with st.container(border=True):
                        st.markdown("### Confusion Matrix")
                        
                        cm = eval_result.confusion_matrix
                        fig_cm = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual"),
                            x=['Non-Sybil', 'Sybil'],
                            y=['Non-Sybil', 'Sybil'],
                            text_auto=True,
                            color_continuous_scale='Blues'
                        )
                        apply_plotly_theme(fig_cm)
                        fig_cm.update_layout(height=320)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with st.expander("Classification Report"):
                        st.code(eval_result.classification_report)
                    
                    with st.container(border=True):
                        st.markdown("### Export Trained Assets")
                        
                        st.caption("Save all required files for inference in the Sybil Detector")
                        
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
                                pyg_data.y = torch.tensor(lr['node_labels'], dtype=torch.long)
                                
                                data_path = f"assets/processed_sybil_data_{timestamp}.pt"
                                torch.save(pyg_data, data_path)
                                st.success(f"Sybil data saved to {data_path}")
                        
                        with col_save3:
                            if st.button("Save Scaler", use_container_width=True):
                                import joblib
                                from utils.clustering_engine import ClusteringEngine
                                
                                engine = ClusteringEngine()
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

"""
Data Exploration Dashboard

Query raw data from BigQuery, construct the 4-layer graph on-the-fly,
and visualize statistical insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from io import BytesIO

from config import MAX_DAYS_RANGE, DATASET_ID
from utils.visualizer import create_legend_html

# Page configuration
st.set_page_config(
    page_title="Data Exploration - Lens Sybil Detector",
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
    .scorecard-delta {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(156, 163, 175, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data_loader():
    """Cache the DataLoader instance."""
    from utils.data_loader import DataLoader
    return DataLoader()


def render_scorecard(label: str, value: str, delta: str = "") -> None:
    """Render a styled scorecard."""
    delta_html = f'<p class="scorecard-delta">{delta}</p>' if delta else ''
    st.markdown(f'''
    <div class="scorecard">
        <p class="scorecard-label">{label}</p>
        <p class="scorecard-value">{value}</p>
        {delta_html}
    </div>
    ''', unsafe_allow_html=True)


def compute_graph_stats(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict:
    """Compute graph statistics."""
    if nodes_df.empty:
        return {
            'total_nodes': 0,
            'total_edges': 0,
            'avg_degree': 0
        }
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes_df['profile_id'].tolist())
    
    if not edges_df.empty:
        for _, row in edges_df.iterrows():
            if row['source'] in G.nodes() and row['target'] in G.nodes():
                G.add_edge(row['source'], row['target'])
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # Combined average degree (in + out) / 2
    avg_in = sum(dict(G.in_degree()).values()) / n_nodes if n_nodes > 0 else 0
    avg_out = sum(dict(G.out_degree()).values()) / n_nodes if n_nodes > 0 else 0
    avg_degree = (avg_in + avg_out) / 2
    
    return {
        'total_nodes': n_nodes,
        'total_edges': n_edges,
        'avg_degree': round(avg_degree, 2)
    }


def render_edge_distribution_chart(edges_df: pd.DataFrame) -> None:
    """Render edge distribution bar chart."""
    if edges_df.empty:
        st.info("No edges to display.")
        return
    
    # Count by layer
    layer_counts = edges_df['layer'].value_counts().reset_index()
    layer_counts.columns = ['Layer', 'Count']
    
    # Count by type
    type_counts = edges_df['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**By Layer**")
        st.bar_chart(layer_counts.set_index('Layer'))
    
    with col2:
        st.markdown("**By Type**")
        st.bar_chart(type_counts.set_index('Type'))


def render_creation_heatmap(nodes_df: pd.DataFrame) -> None:
    """Render account creation frequency chart."""
    if nodes_df.empty or 'created_on' not in nodes_df.columns:
        st.info("No creation time data available.")
        return
    
    # Extract hour from created_on
    nodes_df = nodes_df.copy()
    nodes_df['hour'] = pd.to_datetime(nodes_df['created_on']).dt.hour
    
    hourly_counts = nodes_df['hour'].value_counts().sort_index()
    
    # Fill missing hours with 0
    all_hours = pd.Series(0, index=range(24))
    hourly_counts = all_hours.add(hourly_counts, fill_value=0).astype(int)
    
    st.area_chart(hourly_counts)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


def main():
    """Main application entry point."""
    
    # Title
    st.markdown('<h1 class="page-title">Data Exploration</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Query raw data from BigQuery and explore the 4-layer graph structure</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar - Configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Configuration</p>', unsafe_allow_html=True)
        
        st.caption(f"Dataset: `{DATASET_ID}`")
        st.caption(f"Max date range: {MAX_DAYS_RANGE} days")
        
        st.divider()
        
        # Date selection
        st.markdown('<p class="sidebar-title">Date Range</p>', unsafe_allow_html=True)
        
        today = datetime.now().date()
        default_end = today
        default_start = today - timedelta(days=1)
        
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=today,
            help="Select the start date for data query"
        )
        
        end_date = st.date_input(
            "End Date",
            value=default_end,
            max_value=today,
            help="Select the end date for data query"
        )
        
        # Validate date range
        date_valid = True
        if end_date < start_date:
            st.error("End Date must be >= Start Date")
            date_valid = False
        elif (end_date - start_date).days > MAX_DAYS_RANGE:
            st.error(f"Date range exceeds {MAX_DAYS_RANGE} days limit")
            date_valid = False
        
        st.divider()
        
        # Visualization mode
        st.markdown('<p class="sidebar-title">Visualization</p>', unsafe_allow_html=True)
        
        viz_mode = st.radio(
            "Graph Mode",
            options=["Static (Matplotlib)", "Interactive (PyVis)"],
            index=0,
            help="Choose visualization type"
        )
        
        st.divider()
        
        # Load button (moved below Visualization)
        load_clicked = st.button(
            "Load Data",
            type="primary",
            use_container_width=True,
            disabled=not date_valid
        )
        
        st.divider()
        
        # System info
        st.caption("System Status")
        try:
            loader = get_data_loader()
            if loader.client is not None:
                st.caption("BigQuery: Connected")
            else:
                st.caption("BigQuery: Not connected")
        except Exception as e:
            st.caption(f"BigQuery: Error - {str(e)[:30]}...")
    
    # Handle data loading
    if load_clicked and date_valid:
        try:
            loader = get_data_loader()
            
            with st.spinner("Fetching from BigQuery & Constructing Graph..."):
                # Convert dates to datetime
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.max.time())
                
                result = loader.fetch_and_process_data(start_dt, end_dt)
                
                # Store in session state
                st.session_state['exploration_data'] = result
                st.session_state['exploration_date_range'] = (start_date, end_date)
                
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return
    
    # Check if we have data to display
    if 'exploration_data' not in st.session_state:
        st.info("Select a date range and click 'Load Data' to begin exploration.")
        return
    
    data = st.session_state['exploration_data']
    nodes_df = data['nodes_df']
    features_df = data['features_df']
    edges_df = data['edges_df']
    warnings = data.get('warnings', [])
    
    # Show warnings
    for warning in warnings:
        st.warning(warning)
    
    if nodes_df.empty:
        st.info("No data found for the selected date range.")
        return
    
    # Show date range
    if 'exploration_date_range' in st.session_state:
        s, e = st.session_state['exploration_date_range']
        st.caption(f"Data loaded: {s} to {e}")
    
    st.divider()
    
    # Section 1: Global Stats (Scorecards)
    st.markdown('<p class="section-header">Global Statistics</p>', unsafe_allow_html=True)
    
    stats = compute_graph_stats(nodes_df, edges_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_scorecard("Total Nodes", f"{stats['total_nodes']:,}")
    
    with col2:
        render_scorecard("Total Edges", f"{stats['total_edges']:,}")
    
    with col3:
        render_scorecard("Avg Degree", f"{stats['avg_degree']:.2f}")
    
    st.divider()
    
    # Section 2: Graph Visualization
    st.markdown('<p class="section-header">Network Graph</p>', unsafe_allow_html=True)
    
    # Build NetworkX graph for visualization
    if not edges_df.empty and len(nodes_df) <= 500:
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for _, row in nodes_df.iterrows():
            G.add_node(
                row['profile_id'],
                label=row['handle'][:12] if len(row['handle']) > 12 else row['handle'],
                trust_score=row.get('trust_score', 0)
            )
        
        # Add edges
        for _, row in edges_df.iterrows():
            if row['source'] in G.nodes() and row['target'] in G.nodes():
                G.add_edge(
                    row['source'],
                    row['target'],
                    edge_type=row['layer'],
                    original_type=row['type']
                )
        
        if viz_mode == "Interactive (PyVis)":
            try:
                from pyvis.network import Network
                import tempfile
                import os
                import streamlit.components.v1 as components
                
                # Create PyVis network
                nt = Network(
                    height="500px",
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
                
                components.html(html_content, height=520, scrolling=False)
                
                # Add legend for interactive graph
                st.markdown(create_legend_html(), unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"PyVis rendering failed: {e}. Falling back to static.")
                viz_mode = "Static (Matplotlib)"
        
        if viz_mode == "Static (Matplotlib)":
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
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
    
    elif len(nodes_df) > 500:
        st.info(f"Graph too large ({len(nodes_df)} nodes). Visualization skipped for performance.")
    else:
        st.info("No edges to visualize.")
    
    st.divider()
    
    # Section 3: Data Tables (Tabs)
    st.markdown('<p class="section-header">Data Tables</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Nodes Metadata", "Node Features", "Edges List"])
    
    with tab1:
        metadata_cols = ['profile_id', 'handle', 'display_name', 'bio', 'owned_by', 'trust_score', 'created_on']
        display_cols = [c for c in metadata_cols if c in nodes_df.columns]
        st.dataframe(nodes_df[display_cols], use_container_width=True, height=400)
    
    with tab2:
        st.dataframe(features_df, use_container_width=True, height=400)
    
    with tab3:
        st.dataframe(edges_df, use_container_width=True, height=400)
    
    st.divider()
    
    # Section 4: Deep Insights (Charts)
    st.markdown('<p class="section-header">Deep Insights</p>', unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Edge Distribution**")
        render_edge_distribution_chart(edges_df)
    
    with col_chart2:
        st.markdown("**Account Creation by Hour (UTC)**")
        render_creation_heatmap(nodes_df)
    
    st.divider()
    
    # Section 5: Download Data
    st.markdown('<p class="section-header">Download Data</p>', unsafe_allow_html=True)
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        nodes_csv = convert_df_to_csv(nodes_df)
        st.download_button(
            label="Download Nodes (CSV)",
            data=nodes_csv,
            file_name="nodes_metadata.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        features_csv = convert_df_to_csv(features_df)
        st.download_button(
            label="Download Features (CSV)",
            data=features_csv,
            file_name="node_features.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl3:
        edges_csv = convert_df_to_csv(edges_df)
        st.download_button(
            label="Download Edges (CSV)",
            data=edges_csv,
            file_name="edges_list.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()

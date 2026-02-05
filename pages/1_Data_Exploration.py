"""
Data Exploration Dashboard

Query raw data from BigQuery, construct the 4-layer graph on-the-fly,
and visualize statistical insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import MAX_DAYS_RANGE, DATASET_ID
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
setup_page("Data Exploration")


@st.cache_resource
def get_data_loader():
    """Cache the DataLoader instance."""
    from utils.data_loader import DataLoader
    return DataLoader()


def compute_graph_stats(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict:
    """Compute graph statistics."""
    if nodes_df.empty:
        return {
            'total_nodes': 0,
            'total_edges': 0,
            'avg_degree': 0
        }
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes_df['profile_id'].tolist())
    
    if not edges_df.empty:
        for _, row in edges_df.iterrows():
            if row['source'] in G.nodes() and row['target'] in G.nodes():
                G.add_edge(row['source'], row['target'])
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    avg_in = sum(dict(G.in_degree()).values()) / n_nodes if n_nodes > 0 else 0
    avg_out = sum(dict(G.out_degree()).values()) / n_nodes if n_nodes > 0 else 0
    avg_degree = (avg_in + avg_out) / 2
    
    return {
        'total_nodes': n_nodes,
        'total_edges': n_edges,
        'avg_degree': round(avg_degree, 2)
    }


def render_edge_distribution_chart(edges_df: pd.DataFrame) -> None:
    """Render edge distribution bar chart using Plotly."""
    if edges_df.empty:
        st.info("No edges to display.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        layer_counts = edges_df['layer'].value_counts().reset_index()
        layer_counts.columns = ['Layer', 'Count']
        
        fig = px.bar(
            layer_counts, 
            x='Layer', 
            y='Count',
            color='Layer',
            color_discrete_sequence=[Colors.BLUE, Colors.CYAN, Colors.RED, Colors.PURPLE]
        )
        apply_plotly_theme(fig)
        fig.update_layout(
            title="Edges by Layer",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        type_counts = edges_df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        fig = px.bar(
            type_counts.head(10),
            x='Type',
            y='Count',
            color_discrete_sequence=[Colors.PRIMARY]
        )
        apply_plotly_theme(fig)
        fig.update_layout(
            title="Edges by Type (Top 10)",
            showlegend=False,
            height=300
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


def render_creation_chart(nodes_df: pd.DataFrame) -> None:
    """Render account creation frequency chart using Plotly."""
    if nodes_df.empty or 'created_on' not in nodes_df.columns:
        st.info("No creation time data available.")
        return
    
    df = nodes_df.copy()
    df['hour'] = pd.to_datetime(df['created_on']).dt.hour
    
    hourly_counts = df['hour'].value_counts().sort_index()
    all_hours = pd.Series(0, index=range(24))
    hourly_counts = all_hours.add(hourly_counts, fill_value=0).astype(int)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_counts.index,
        y=hourly_counts.values,
        fill='tozeroy',
        line=dict(color=Colors.PRIMARY, width=2),
        fillcolor='rgba(37, 99, 235, 0.1)'
    ))
    
    apply_plotly_theme(fig)
    fig.update_layout(
        title="Account Creation by Hour (UTC)",
        xaxis_title="Hour",
        yaxis_title="Count",
        height=300
    )
    fig.update_xaxes(dtick=2)
    
    st.plotly_chart(fig, use_container_width=True)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


def main():
    """Main application entry point."""
    
    # Header
    page_header(
        "Data Exploration",
        "Query raw data from BigQuery and explore the 4-layer graph structure"
    )
    
    # Sidebar
    with st.sidebar:
        sidebar_header("Configuration")
        
        st.caption(f"Dataset: `{DATASET_ID}`")
        st.caption(f"Max range: {MAX_DAYS_RANGE} days")
        
        st.divider()
        
        sidebar_header("Date Range")
        
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
        
        date_valid = True
        if end_date < start_date:
            st.error("End date must be after start date")
            date_valid = False
        elif (end_date - start_date).days > MAX_DAYS_RANGE:
            st.error(f"Range exceeds {MAX_DAYS_RANGE} days limit")
            date_valid = False
        
        st.divider()
        
        sidebar_header("Visualization")
        
        viz_mode = st.radio(
            "Graph Mode",
            options=["Static (Matplotlib)", "Interactive (PyVis)"],
            index=0,
            help="Choose visualization type"
        )
        
        st.divider()
        
        load_clicked = st.button(
            "Load Data",
            type="primary",
            use_container_width=True,
            disabled=not date_valid
        )
        
        st.divider()
        
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
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.max.time())
                
                result = loader.fetch_and_process_data(start_dt, end_dt)
                
                st.session_state['exploration_data'] = result
                st.session_state['exploration_date_range'] = (start_date, end_date)
                
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return
    
    # Check for data
    if 'exploration_data' not in st.session_state:
        st.info("Select a date range and click 'Load Data' to begin exploration.")
        return
    
    data = st.session_state['exploration_data']
    nodes_df = data['nodes_df'].copy()
    edges_df = data['edges_df']
    warnings = data.get('warnings', [])
    
    for warning in warnings:
        st.warning(warning)
    
    if nodes_df.empty:
        st.info("No data found for the selected date range.")
        return
    
    if 'exploration_date_range' in st.session_state:
        s, e = st.session_state['exploration_date_range']
        st.caption(f"Data loaded: {s} to {e}")
    
    st.divider()
    
    # Section 1: Global Stats
    section_header("Global Statistics")
    
    stats = compute_graph_stats(nodes_df, edges_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card("Total Nodes", f"{stats['total_nodes']:,}", compact=True)
    
    with col2:
        metric_card("Total Edges", f"{stats['total_edges']:,}", compact=True)
    
    with col3:
        metric_card("Avg Degree", f"{stats['avg_degree']:.2f}", compact=True)
    
    st.divider()
    
    # Section 2: Network Graph
    section_header("Network Graph")
    
    if not edges_df.empty and len(nodes_df) <= 500:
        G = nx.DiGraph()
        
        for _, row in nodes_df.iterrows():
            G.add_node(
                row['profile_id'],
                label=row['handle'][:12] if len(row['handle']) > 12 else row['handle'],
                trust_score=row.get('trust_score', 0)
            )
        
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
                
                nt = Network(
                    height="500px",
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
                
                components.html(html_content, height=520, scrolling=False)
                st.markdown(create_legend_html(), unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"PyVis rendering failed: {e}. Falling back to static.")
                viz_mode = "Static (Matplotlib)"
        
        if viz_mode == "Static (Matplotlib)":
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
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
    
    elif len(nodes_df) > 500:
        st.info(f"Graph too large ({len(nodes_df)} nodes). Visualization skipped for performance.")
    else:
        st.info("No edges to visualize.")
    
    st.divider()
    
    # Section 3: Data Tables
    section_header("Data Tables")
    
    tab1, tab2 = st.tabs(["Nodes", "Edges"])
    
    with tab1:
        st.dataframe(nodes_df, use_container_width=True, height=400, hide_index=True)
    
    with tab2:
        st.dataframe(edges_df, use_container_width=True, height=400, hide_index=True)
    
    st.divider()

    # Section 4: Download
    section_header("Download Data")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        nodes_csv = convert_df_to_csv(nodes_df)
        st.download_button(
            label="Download Nodes Data (.csv)",
            data=nodes_csv,
            file_name=f"nodes_{start_date}-{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        edges_csv = convert_df_to_csv(edges_df)
        st.download_button(
            label="Download Edges Data (.csv)",
            data=edges_csv,
            file_name=f"edges_{start_date}-{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.divider()
    
    # Section 5: Deep Insights
    section_header("Deep Insights")
    
    # col_chart1, col_chart2 = st.columns(2)
    
    # with col_chart1:
    with st.container(border=True):
        render_edge_distribution_chart(edges_df)
    
    # with col_chart2:
    with st.container(border=True):
        render_creation_chart(nodes_df)
    
    st.divider()
    
    # Section 6: Continue
    section_header("Next Step")
    
    with st.container(border=True):
        st.markdown("### Continue to Model Laboratory")
        st.caption("Use this dataset to train a Sybil detection model")
        
        if st.button(
            "Continue to Model Laboratory",
            type="primary",
            use_container_width=True
        ):
            st.switch_page("pages/2_Model_Laboratory.py")


if __name__ == "__main__":
    main()

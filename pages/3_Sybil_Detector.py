"""
Sybil Detector - Model Inference

A GNN-based identity verification system for detecting Sybil accounts 
on the Lens Protocol network.
"""

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from utils.predictor import SybilPredictor
from utils.data_fetcher import bq_fetcher, mock_bq_fetcher
from utils.visualizer import (
    build_analysis_graph,
    visualize_interactive_graph,
    visualize_static_graph,
    create_legend_html
)
from utils.ui import (
    setup_page,
    page_header,
    sidebar_header,
    metric_card,
    profile_id_badge
)


# Page setup
setup_page("Sybil Detector")


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load and cache the SybilPredictor instance."""
    return SybilPredictor()


def _render_results(result, G, has_edges):
    """Render analysis results and network graph from session state data."""
    
    st.divider()
    
    # Results container
    with st.container(border=True):
        st.markdown("### Analysis Results")
        
        col_metrics, col_details = st.columns([1, 2])
        
        with col_metrics:
            # Profile Info
            st.markdown("##### Profile Information")
            
            st.markdown(f'<p class="small-label">Profile ID</p>', unsafe_allow_html=True)
            profile_id_badge(result["profile_id"])
            
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
            st.markdown(f'<p class="small-label">Handle</p>', unsafe_allow_html=True)
            st.markdown(f"**@{result['handle']}**")
            
            st.divider()
            
            # Verdict
            st.markdown("##### Prediction")
            
            if result["prediction"] == "SYBIL":
                st.markdown(f'<p class="verdict-sybil">SYBIL</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="verdict-nonsybil">NON-SYBIL</p>', unsafe_allow_html=True)
            
            st.divider()
            
            # Metrics
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                metric_card(
                    "Confidence",
                    result["sybil_probability_formatted"],
                    compact=True
                )
            
            with col_m2:
                risk_level = result["analysis"]["risk_level"]
                risk_status = "danger" if risk_level == "High" else ("warning" if risk_level == "Medium" else "safe")
                metric_card(
                    "Risk Level",
                    risk_level,
                    status=risk_status,
                    compact=True
                )
            
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
            metric_card(
                "Edges Found",
                str(result["analysis"]["edges_found"]),
                compact=True
            )
        
        with col_details:
            st.markdown("##### Network Graph")
            
            if not has_edges:
                st.info("No connections found in reference graph.")
            else:
                # Visualization mode selection
                viz_mode = st.radio(
                    "Graph Visualization Mode",
                    options=["Static (Matplotlib)", "Interactive (PyVis)"],
                    index=0,
                    horizontal=True,
                    help="Choose visualization type",
                    key="sybil_viz_mode"
                )
                
                # Render graph using shared visualization functions
                with st.container(border=True):
                    st.markdown("### Network Analysis")
                    
                    if viz_mode == "Interactive (PyVis)":
                        try:
                            html_content = visualize_interactive_graph(G, is_classify=True)
                            components.html(html_content, height=760, scrolling=False)
                            st.markdown(create_legend_html(is_classify=True), unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"PyVis rendering failed: {e}. Falling back to static.")
                            viz_mode = "Static (Matplotlib)"
                    
                    if viz_mode == "Static (Matplotlib)":
                        fig = visualize_static_graph(G, is_classify=True)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)


def main():
    """Main application entry point."""
    
    # Initialize session state
    if "sybil_result" not in st.session_state:
        st.session_state.sybil_result = None
        st.session_state.sybil_graph = None
        st.session_state.sybil_has_edges = False
    
    # Sidebar
    with st.sidebar:
        sidebar_header("Configuration")
        
        data_source = st.radio(
            "Data Source",
            options=["Mock Data", "Real Data (BigQuery)"],
            index=0,
            help="Select data source for profile lookup"
        )
        
        st.divider()
        
        st.caption("System Status")
        
        try:
            with st.spinner("Loading model..."):
                predictor = load_predictor()
            st.caption("Model: Loaded")
            st.caption("Scaler: Loaded")
            st.caption(f"Reference nodes: {predictor.num_ref_nodes}")
            model_loaded = True
        except Exception as e:
            st.caption("Model: Not loaded")
            st.caption(f"Error: {str(e)[:50]}...")
            model_loaded = False
    
    # Header
    page_header(
        "Sybil Detector",
        "GNN-based identity verification for Lens Protocol"
    )
    
    # Input section
    with st.container(border=True):
        st.markdown("### Profile Analysis")
        
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            profile_id = st.text_input(
                "Target Profile ID",
                placeholder="Enter profile ID (e.g., 0x...)",
                label_visibility="collapsed"
            )
        
        with col_button:
            analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    
    # Run analysis when button is clicked — persist results to session state
    if analyze_clicked and profile_id:
        if not model_loaded:
            st.error("Model not loaded. Please check the system status in the sidebar.")
            return
        
        fetcher = mock_bq_fetcher if data_source == "Mock Data" else bq_fetcher
        
        with st.spinner("Analyzing profile..."):
            try:
                result, new_edges, new_types, new_dirs = predictor.predict(profile_id, fetcher)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.sybil_result = None
                return
        
        if "error" in result:
            st.error(f"Profile not found: {profile_id}")
            st.session_state.sybil_result = None
            return
        
        # Build graph and save everything to session state
        has_edges = new_edges.numel() > 0
        G = None
        if has_edges:
            G, _, _ = build_analysis_graph(
                result["node_info"],
                new_edges,
                new_types,
                new_dirs,
                predictor.df_ref,
                result,
                ref_labels=predictor.ref_data.y
            )
        
        st.session_state.sybil_result = result
        st.session_state.sybil_graph = G
        st.session_state.sybil_has_edges = has_edges
    
    elif analyze_clicked and not profile_id:
        st.warning("Please enter a Profile ID to analyze.")
    
    # Render results from session state (survives reruns from widget interactions)
    if st.session_state.sybil_result is not None:
        _render_results(
            st.session_state.sybil_result,
            st.session_state.sybil_graph,
            st.session_state.sybil_has_edges
        )
    elif not analyze_clicked:
        # Show placeholder when no analysis has been run
        with st.container(border=True):
            st.markdown("### How to Use")
            st.markdown("""
            1. Enter a **Profile ID** in the input field above
            2. Click **Analyze** to run the GNN-based prediction
            3. View the **prediction result**, **confidence score**, and **network visualization**
            
            The model analyzes:
            - Follow relationships
            - Interaction patterns (comments, likes, tips)
            - Co-ownership connections (shared wallets)
            - Profile similarity (handles, bios, avatars)
            """)


if __name__ == "__main__":
    main()

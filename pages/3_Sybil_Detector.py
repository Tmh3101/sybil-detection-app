"""
Sybil Detector - Model Inference

A GNN-based identity verification system for detecting Sybil accounts 
on the Lens Protocol network.
"""

import streamlit as st
import streamlit.components.v1 as components

from utils.predictor import SybilPredictor
from utils.data_fetcher import bq_fetcher, mock_bq_fetcher
from utils.visualizer import (
    render_static_graph,
    render_interactive_graph,
    create_legend_html,
    get_debug_messages,
    PYVIS_AVAILABLE
)
from utils.ui import (
    setup_page,
    page_header,
    section_header,
    sidebar_header,
    metric_card,
    profile_id_badge,
    risk_badge,
    Colors
)


# Page setup
setup_page("Sybil Detector")


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load and cache the SybilPredictor instance."""
    return SybilPredictor()


def render_graph_section(
    result: dict,
    new_edges,
    new_types: list,
    new_dirs: list,
    df_ref,
    viz_mode: str,
    ref_labels=None
) -> None:
    """Render the network visualization based on selected mode."""
    st.markdown("### Network Analysis")
    
    if new_edges.numel() == 0:
        st.info("No connections found in reference graph.")
        return
    
    use_interactive = viz_mode == "Interactive (PyVis)"
    fallback_used = False
    
    if use_interactive:
        error_message = None
        try:
            html_path = render_interactive_graph(
                result["node_info"],
                new_edges,
                new_types,
                new_dirs,
                df_ref,
                result,
                ref_labels=ref_labels
            )
            
            if html_path is not None:
                with open(html_path, 'r', encoding='utf-8') as f:
                    graph_html = f.read()
                
                if len(graph_html) > 0:
                    components.html(graph_html, height=570, scrolling=False)
                    st.markdown(create_legend_html(), unsafe_allow_html=True)
                    return
                else:
                    error_message = "Generated HTML file is empty"
                    fallback_used = True
            else:
                error_message = "render_interactive_graph returned None"
                fallback_used = True
        except Exception as e:
            error_message = str(e)
            fallback_used = True
        
        if fallback_used:
            warning_msg = "Switched to static mode due to rendering issue."
            if error_message:
                warning_msg += f" Error: {error_message}"
            st.warning(warning_msg)
            
            debug_msgs = get_debug_messages()
            if debug_msgs:
                with st.expander("Debug Information", expanded=False):
                    for msg in debug_msgs:
                        if "[ERROR]" in msg:
                            st.error(msg)
                        else:
                            st.text(msg)
    
    # Static mode (or fallback)
    fig = render_static_graph(
        result["node_info"],
        new_edges,
        new_types,
        new_dirs,
        df_ref,
        result,
        ref_labels=ref_labels
    )
    
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Node is isolated - no connections found in reference graph.")


def main():
    """Main application entry point."""
    
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
        
        sidebar_header("Visualization")
        
        viz_options = ["Interactive (PyVis)", "Static (Matplotlib)"]
        if not PYVIS_AVAILABLE:
            viz_options = ["Static (Matplotlib)"]
            st.caption("PyVis not available")
        
        viz_mode = st.radio(
            "Graph Mode",
            options=viz_options,
            index=0,
            help="Interactive mode allows zooming and panning"
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
    
    # Analysis section
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
                return
        
        if "error" in result:
            st.error(f"Profile not found: {profile_id}")
            return
        
        st.divider()
        
        # Results container
        with st.container(border=True):
            st.markdown("### Analysis Results")
            
            col_metrics, col_graph = st.columns([1, 2])
            
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
            
            with col_graph:
                render_graph_section(
                    result,
                    new_edges,
                    new_types,
                    new_dirs,
                    predictor.df_ref,
                    viz_mode,
                    ref_labels=predictor.ref_data.y
                )
    
    elif analyze_clicked and not profile_id:
        st.warning("Please enter a Profile ID to analyze.")
    
    else:
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

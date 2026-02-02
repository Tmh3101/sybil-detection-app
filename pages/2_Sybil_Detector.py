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


# Page configuration
st.set_page_config(
    page_title="Sybil Detector - Lens Protocol",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimal, professional styling
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
    
    /* Verdict styling */
    .verdict-sybil {
        font-size: 2rem;
        font-weight: 700;
        color: #dc2626 !important;
        margin: 0;
    }
    .verdict-nonsybil {
        font-size: 2rem;
        font-weight: 700;
        color: #16a34a !important;
        margin: 0;
    }
    .verdict-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    /* Metric styling */
    .metric-container {
        margin-bottom: 1.5rem;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-value-secondary {
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.3;
    }
    
    /* Code/monospace for IDs */
    .profile-id {
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.875rem;
        background-color: rgba(156, 163, 175, 0.2);
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
    
    /* Risk level colors */
    .risk-high {
        color: #dc2626 !important;
        font-weight: 600;
    }
    .risk-medium {
        color: #d97706 !important;
        font-weight: 600;
    }
    .risk-low {
        color: #16a34a !important;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(156, 163, 175, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load and cache the SybilPredictor instance."""
    return SybilPredictor()


def get_risk_class(risk_level: str) -> str:
    """Return CSS class for risk level."""
    return {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low"
    }.get(risk_level, "")


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
    st.markdown('<p class="metric-label">Network Analysis</p>', unsafe_allow_html=True)
    
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
    
    # Sidebar - Configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Configuration</p>', unsafe_allow_html=True)
        
        data_source = st.radio(
            "Data Source",
            options=["Mock Data", "Real Data (BigQuery)"],
            index=0,
            help="Select data source for profile lookup"
        )
        
        st.divider()
        
        # Visualization settings
        st.markdown('<p class="sidebar-title">Visualization</p>', unsafe_allow_html=True)
        
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
        
        # System status
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
    
    # Main content area
    st.markdown('<h1 class="page-title">Sybil Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">GNN-based identity verification for Lens Protocol</p>',
        unsafe_allow_html=True
    )
    
    # Input section
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        profile_id = st.text_input(
            "Target Profile ID",
            placeholder="0x...",
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
            col_metrics, col_graph = st.columns([1, 2])
            
            with col_metrics:
                st.markdown('<p class="metric-label">Profile ID</p>', unsafe_allow_html=True)
                st.markdown(f'<code class="profile-id">{result["profile_id"]}</code>', unsafe_allow_html=True)

                st.markdown('<p class="metric-label">Handle</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value-secondary">@{result["handle"]}</p>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown('<p class="verdict-label">Prediction</p>', unsafe_allow_html=True)
                if result["prediction"] == "SYBIL":
                    st.markdown('<p class="verdict-sybil">SYBIL</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="verdict-nonsybil">NON-SYBIL</p>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown('<p class="metric-label">Confidence Score</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="metric-value">{result["sybil_probability_formatted"]}</p>', 
                    unsafe_allow_html=True
                )
                
                risk_level = result["analysis"]["risk_level"]
                risk_class = get_risk_class(risk_level)
                st.markdown('<p class="metric-label">Risk Level</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="metric-value-secondary {risk_class}">{risk_level}</p>', 
                    unsafe_allow_html=True
                )
                
                st.markdown('<p class="metric-label">Edges Found</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="metric-value-secondary">{result["analysis"]["edges_found"]}</p>', 
                    unsafe_allow_html=True
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


if __name__ == "__main__":
    main()

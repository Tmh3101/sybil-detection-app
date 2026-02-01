"""
Lens Protocol Sybil Detector

A GNN-based identity verification system for detecting Sybil accounts 
on the Lens Protocol network.
"""

import streamlit as st

from utils.predictor import SybilPredictor
from utils.data_fetcher import bq_fetcher, mock_bq_fetcher
from utils.visualizer import visualize_prediction_graph


# Page configuration
st.set_page_config(
    page_title="Lens Protocol Sybil Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimal, professional styling
st.markdown("""
<style>
    /* Typography */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    .subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
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
    
    /* Container adjustments */
    .stContainer {
        padding: 1rem;
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
        
        # System status
        st.caption("System Status")
        
        # Load predictor
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
    st.markdown('<h1 class="main-title">Lens Sybil Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">GNN-based identity verification system</p>', unsafe_allow_html=True)
    
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
        
        # Select data fetcher
        fetcher = mock_bq_fetcher if data_source == "Mock Data" else bq_fetcher
        
        # Run prediction
        with st.spinner("Analyzing profile..."):
            try:
                result, new_edges, new_types = predictor.predict(profile_id, fetcher)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return
        
        # Check for errors
        if "error" in result:
            st.error(f"Profile not found: {profile_id}")
            return
        
        st.divider()
        
        # Results container
        with st.container(border=True):
            col_metrics, col_graph = st.columns([1, 2])
            
            # Left column - Verdict and metrics
            with col_metrics:
                # Profile info
                st.markdown('<p class="metric-label">Handle</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value-secondary">@{result["handle"]}</p>', unsafe_allow_html=True)
                
                st.markdown('<p class="metric-label">Profile ID</p>', unsafe_allow_html=True)
                st.markdown(f'<code class="profile-id">{result["profile_id"][:20]}...</code>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Prediction verdict
                st.markdown('<p class="verdict-label">Prediction</p>', unsafe_allow_html=True)
                if result["prediction"] == "SYBIL":
                    st.markdown('<p class="verdict-sybil">SYBIL</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="verdict-nonsybil">NON-SYBIL</p>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Confidence score
                st.markdown('<p class="metric-label">Confidence Score</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="metric-value">{result["sybil_probability_formatted"]}</p>', 
                    unsafe_allow_html=True
                )
                
                # Risk level
                risk_level = result["analysis"]["risk_level"]
                risk_class = get_risk_class(risk_level)
                st.markdown('<p class="metric-label">Risk Level</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="metric-value-secondary {risk_class}">{risk_level}</p>', 
                    unsafe_allow_html=True
                )
                
                # Edge analysis
                st.markdown('<p class="metric-label">Edges Found</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="metric-value-secondary">{result["analysis"]["edges_found"]}</p>', 
                    unsafe_allow_html=True
                )
                
                # Co-owner flag
                if result["analysis"]["has_co_owner"]:
                    st.markdown('<p class="metric-label">Co-owner Detected</p>', unsafe_allow_html=True)
                    st.markdown(
                        '<p class="metric-value-secondary risk-high">Yes</p>', 
                        unsafe_allow_html=True
                    )
            
            # Right column - Network visualization
            with col_graph:
                st.markdown('<p class="metric-label">Network Analysis</p>', unsafe_allow_html=True)
                
                if new_edges.numel() > 0:
                    fig = visualize_prediction_graph(
                        result["node_info"],
                        new_edges,
                        new_types,
                        predictor.df_ref,
                        result
                    )
                    
                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Node is isolated - no connections found in reference graph.")
                else:
                    st.info("No connections found in reference graph.")
    
    elif analyze_clicked and not profile_id:
        st.warning("Please enter a Profile ID to analyze.")


if __name__ == "__main__":
    main()

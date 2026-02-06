"""
Lens Protocol Sybil Detection App

Home page with navigation cards to different modules.
"""

import streamlit as st
from utils.ui import load_css, feature_card


# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Lens Sybil Detection App",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load global CSS
load_css()


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<h1 class="page-title">Lens Sybil Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">'
        'A toolkit for detecting Sybil accounts on Lens Protocol using Graph Neural Networks.'
        '</p>',
        unsafe_allow_html=True
    )
    
    # Navigation cards - 3 columns
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        feature_card(
            title="Data Exploration",
            description="Query data from BigQuery and construct the 4-layer graph.",
            features=[
                "Date range selection",
                "Graph construction",
                "Interactive visualization",
                "CSV export"
            ],
            tag="ETL & Analytics"
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Open Data Exploration", key="btn_exploration", use_container_width=True):
            st.switch_page("pages/1_Data_Exploration.py")
    
    with col2:
        feature_card(
            title="Model Laboratory",
            description="Clustering, labeling, and GAT model training workbench.",
            features=[
                "K-Means clustering",
                "Rule-based labeling",
                "GAT hyperparameters",
                "Live training metrics"
            ],
            tag="ML Training"
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Open Model Laboratory", key="btn_laboratory", use_container_width=True):
            st.switch_page("pages/2_Model_Laboratory.py")
    
    with col3:
        feature_card(
            title="Sybil Detector",
            description="Analyze profiles using trained GAT model in real-time.",
            features=[
                "Profile ID lookup",
                "GNN-based prediction",
                "Confidence scoring",
                "Network visualization"
            ],
            tag="Model Inference"
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Open Sybil Detector", key="btn_detector", use_container_width=True):
            st.switch_page("pages/3_Sybil_Detector.py")
    
    # Footer
    st.divider()
    st.markdown(
        '<p style="text-align: center; color: var(--color-text-muted); font-size: 0.8125rem;">'
        'Built with Streamlit, PyTorch Geometric, and NetworkX'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: var(--color-text-muted); font-size: 0.6875rem;">'
        'Lens Protocol Sybil Detection App v1.0'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

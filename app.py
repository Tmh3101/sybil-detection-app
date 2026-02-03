"""
Lens Protocol Sybil Detection App

Home page with navigation cards to different modules.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Lens Sybil Detection App",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    /* Typography */
    .app-title {
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-subtitle {
        font-size: 1.125rem;
        color: #9ca3af;
        margin-bottom: 3rem;
        max-width: 600px;
    }
    
    /* Card styling */
    .nav-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        min-height: 280px;
        display: flex;
        flex-direction: column;
    }
    .nav-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
    }
    .card-description {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.6;
        flex-grow: 1;
    }
    .card-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        color: #3b82f6;
        background: rgba(59, 130, 246, 0.15);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin-top: 1rem;
    }
    .card-tag-purple {
        color: #8b5cf6;
        background: rgba(139, 92, 246, 0.15);
    }
    
    /* Feature list */
    .feature-list {
        margin-top: 0.75rem;
        padding-left: 0;
    }
    .feature-item {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 0.375rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .feature-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #3b82f6;
        flex-shrink: 0;
    }
    .feature-dot-purple {
        background: #8b5cf6;
    }
    
    /* Footer */
    .app-footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        text-align: center;
        color: #64748b;
        font-size: 0.875rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button override */
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def render_card(
    title: str,
    description: str,
    features: list,
    tag: str,
    tag_color: str = "blue",
    icon: str = ""
) -> None:
    """Render a navigation card."""
    tag_class = "card-tag-purple" if tag_color == "purple" else ""
    dot_class = "feature-dot-purple" if tag_color == "purple" else ""
    
    features_html = "".join([
        f'<div class="feature-item"><div class="feature-dot {dot_class}"></div>{f}</div>'
        for f in features
    ])
    
    st.markdown(f'''
    <div class="nav-card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-description">{description}</div>
        <div class="feature-list">{features_html}</div>
        <span class="card-tag {tag_class}">{tag}</span>
    </div>
    ''', unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<h1 class="app-title">Lens Sybil Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">'
        'A comprehensive toolkit for detecting Sybil accounts on Lens Protocol '
        'using Graph Neural Networks and multi-layer relationship analysis.'
        '</p>',
        unsafe_allow_html=True
    )
    
    # Navigation cards
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        render_card(
            title="Data Exploration",
            description="Query raw data from BigQuery, construct the 4-layer graph on-the-fly, and visualize statistical insights.",
            features=[
                "Date range selection",
                "4-layer graph construction",
                "Interactive visualization",
                "CSV data export"
            ],
            tag="ETL & Analytics",
            tag_color="blue",
            icon="📊"
        )
        
        if st.button("Open Data Exploration", key="btn_exploration", use_container_width=True):
            st.switch_page("pages/1_Data_Exploration.py")
    
    with col2:
        render_card(
            title="Model Laboratory",
            description="Scientific workbench for clustering, semi-supervised labeling, and GAT model training on custom data.",
            features=[
                "K-Means clustering",
                "Rule-based labeling",
                "GAT hyperparameter tuning",
                "Live training metrics"
            ],
            tag="ML Training",
            tag_color="blue",
            icon="🧪"
        )
        
        if st.button("Open Model Laboratory", key="btn_laboratory", use_container_width=True):
            st.switch_page("pages/2_Model_Laboratory.py")
    
    with col3:
        render_card(
            title="Sybil Detector",
            description="Analyze individual profiles using a trained GAT model to detect potential Sybil accounts in real-time.",
            features=[
                "Profile ID lookup",
                "GNN-based prediction",
                "Confidence scoring",
                "Network visualization"
            ],
            tag="Model Inference",
            tag_color="purple",
            icon="🔍"
        )
        
        if st.button("Open Sybil Detector", key="btn_detector", use_container_width=True):
            st.switch_page("pages/3_Sybil_Detector.py")
    
    # Footer
    st.markdown('''
    <div class="app-footer">
        <p>Built with Streamlit, PyTorch Geometric, and NetworkX</p>
        <p style="margin-top: 0.5rem; font-size: 0.75rem; color: #475569;">
            Lens Protocol Sybil Detection App v1.0
        </p>
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

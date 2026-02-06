"""
UI Utility Module - Sybil Detection App

Provides consistent styling and layout helpers following the 
Scientific Workbench design system.
"""

import os
import streamlit as st
from typing import Optional, Literal


# Path to the global CSS file
CSS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")


def load_css() -> None:
    """
    Load and inject the global CSS stylesheet.
    
    Reads assets/style.css and injects it via st.markdown.
    Should be called once at the top of each page.
    """
    try:
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Global stylesheet not found. Using default styles.")


def setup_page(
    page_title: str,
    layout: Literal["centered", "wide"] = "wide",
    initial_sidebar_state: Literal["auto", "expanded", "collapsed"] = "expanded"
) -> None:
    """
    Initialize page configuration and load global styles.
    
    This should be called at the very top of every page file.
    
    Args:
        page_title: The page title shown in browser tab.
        layout: Page layout ("centered" or "wide").
        initial_sidebar_state: Initial sidebar state.
    """
    st.set_page_config(
        page_title=f"{page_title} - Lens Sybil Detector",
        page_icon=None,  # No emojis per design system
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )
    
    # Load global CSS
    load_css()


def page_header(title: str, subtitle: Optional[str] = None) -> None:
    """
    Render a consistent page header.
    
    Args:
        title: Main page title.
        subtitle: Optional subtitle/description.
    """
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def section_header(title: str) -> None:
    """
    Render a section header with bottom border.
    
    Args:
        title: Section title text.
    """
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)


def sidebar_header(title: str) -> None:
    """
    Render a sidebar section header.
    
    Args:
        title: Section title for sidebar.
    """
    st.markdown(f'<p class="sidebar-title">{title}</p>', unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    status: Optional[Literal["safe", "danger", "warning", "neutral"]] = None,
    compact: bool = False
) -> None:
    """
    Render a styled metric card.
    
    Creates a dashboard-style metric widget with label, value,
    and optional delta/status indicator.
    
    Args:
        label: Metric label (uppercase, small).
        value: Main metric value (large, monospace).
        delta: Optional secondary text below value.
        status: Optional status color for the value.
        compact: If True, uses smaller padding/font.
    """
    status_class = f"status-{status}" if status else ""
    delta_html = f'<p class="delta">{delta}</p>' if delta else ""
    
    font_size = "1.125rem" if compact else "1.25rem"
    
    st.markdown(f'''
    <div class="metric-card" style="margin: 1rem">
        <p class="label">{label}</p>
        <p class="value {status_class}" style="font-size: {font_size};">{value}</p>
        {delta_html}
    </div>
    ''', unsafe_allow_html=True)


def caption(text: str) -> None:
    """
    Render muted caption text.
    
    Args:
        text: Caption text to display.
    """
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)


def small_label(text: str) -> None:
    """
    Render a small uppercase label.
    
    Args:
        text: Label text.
    """
    st.markdown(f'<p class="small-label">{text}</p>', unsafe_allow_html=True)


def verdict_badge(prediction: str, confidence: str) -> None:
    """
    Render a prediction verdict badge.
    
    Args:
        prediction: "SYBIL" or "NON-SYBIL".
        confidence: Confidence percentage string.
    """
    verdict_class = "verdict-sybil" if prediction == "SYBIL" else "verdict-nonsybil"
    
    st.markdown(f'''
    <p class="small-label">Prediction</p>
    <p class="{verdict_class}">{prediction}</p>
    ''', unsafe_allow_html=True)


def risk_badge(risk_level: str) -> str:
    """
    Get CSS class for risk level.
    
    Args:
        risk_level: "High", "Medium", or "Low".
        
    Returns:
        CSS class name for the risk level.
    """
    return {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low"
    }.get(risk_level, "")


def profile_id_badge(profile_id: str) -> None:
    """
    Render a profile ID in monospace badge style.
    
    Args:
        profile_id: The profile ID to display.
    """
    st.markdown(f'<code class="profile-id">{profile_id}</code>', unsafe_allow_html=True)


def status_indicator(
    label: str,
    status: Literal["success", "warning", "error", "neutral"],
    text: str
) -> None:
    """
    Render a status indicator with colored text.
    
    Args:
        label: Status label.
        status: Status type for coloring.
        text: Status text.
    """
    color_map = {
        "success": "var(--color-safe)",
        "warning": "var(--color-warning)",
        "error": "var(--color-danger)",
        "neutral": "var(--color-neutral)"
    }
    color = color_map.get(status, color_map["neutral"])
    
    st.markdown(f'''
    <p style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 0.25rem;">
        {label}
    </p>
    <p style="font-size: 0.875rem; color: {color}; font-weight: 500;">
        {text}
    </p>
    ''', unsafe_allow_html=True)


def card_header(title: str, subtitle: Optional[str] = None) -> None:
    """
    Render a header inside a card container.
    
    Args:
        title: Card title.
        subtitle: Optional subtitle.
    """
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


# Plotly chart theme configuration
PLOTLY_THEME = {
    "template": "plotly_white",
    "color_discrete_sequence": [
        "#2563EB",  # Lens Blue
        "#059669",  # Scientific Teal
        "#DC2626",  # Alert Red
        "#7C3AED",  # Purple
        "#06B6D4",  # Cyan
        "#D97706",  # Amber
    ],
    "layout": {
        "font": {"family": "Inter, sans-serif", "color": "#111827"},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "title": {"font": {"size": 14, "color": "#111827"}},
        "xaxis": {
            "gridcolor": "#F3F4F6",
            "linecolor": "#E5E7EB",
            "tickfont": {"size": 11, "color": "#6B7280"},
        },
        "yaxis": {
            "gridcolor": "#F3F4F6",
            "linecolor": "#E5E7EB",
            "tickfont": {"size": 11, "color": "#6B7280"},
        },
        "legend": {"font": {"size": 11, "color": "#6B7280"}},
    },
}


def apply_plotly_theme(fig) -> None:
    """
    Apply the design system theme to a Plotly figure.
    
    Args:
        fig: Plotly figure object to style.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", color="#111827"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            gridcolor="#F3F4F6",
            linecolor="#E5E7EB",
            tickfont=dict(size=11, color="#6B7280"),
        ),
        yaxis=dict(
            gridcolor="#F3F4F6",
            linecolor="#E5E7EB",
            tickfont=dict(size=11, color="#6B7280"),
        ),
        legend=dict(font=dict(size=11, color="#6B7280")),
    )


# Color constants for easy access
class Colors:
    """Design system color constants."""
    
    # Backgrounds
    BG_PRIMARY = "#FFFFFF"
    BG_SECONDARY = "#F8F9FA"
    SIDEBAR_BG = "#111827"       # Dark sidebar
    
    # Borders
    BORDER = "#E5E7EB"
    BORDER_HOVER = "#D1D5DB"
    SIDEBAR_BORDER = "#1F2937"
    
    # Text
    TEXT_PRIMARY = "#111827"
    TEXT_SECONDARY = "#6B7280"
    TEXT_MUTED = "#9CA3AF"
    SIDEBAR_TEXT = "#E5E7EB"     # Light text for dark sidebar
    
    # Actions
    PRIMARY = "#2563EB"
    PRIMARY_HOVER = "#1D4ED8"
    
    # Status
    SAFE = "#059669"
    DANGER = "#DC2626"
    WARNING = "#D97706"
    NEUTRAL = "#4B5563"
    
    # Chart colors
    BLUE = "#3B82F6"
    CYAN = "#06B6D4"
    PURPLE = "#7C3AED"
    TEAL = "#059669"
    RED = "#DC2626"


def feature_card(
    title: str,
    description: str,
    features: list,
    tag: str
) -> None:
    """
    Render a compact feature card with accent border.
    
    Used for navigation cards on the home page.
    
    Args:
        title: Card title.
        description: Short description.
        features: List of feature bullet points.
        tag: Category tag (e.g., "ETL & Analytics").
    """
    features_html = "".join([
        f'<li style="font-size: 0.8125rem; color: #6B7280; margin-bottom: 0.25rem;">{f}</li>'
        for f in features
    ])
    
    st.markdown(f'''
    <div class="feature-card">
        <p class="card-title">{title}</p>
        <p class="card-description">{description}</p>
        <ul style="margin: 0.5rem 0; padding-left: 1.25rem; list-style-type: disc;">
            {features_html}
        </ul>
        <span class="card-tag">{tag}</span>
    </div>
    ''', unsafe_allow_html=True)

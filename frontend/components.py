from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def inject_css(css: str) -> None:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(
    eyebrow: str,
    title: str,
    description: str,
    badge: str | None = None,
) -> None:
    badge_html = f'<span class="page-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="eyebrow">{eyebrow}</div>
            <div class="page-title-row">
                <h1>{title}</h1>
                {badge_html}
            </div>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div>
                <h2>{title}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    helper: str = "",
    accent: str = "violet",
) -> None:
    st.markdown(
        f"""
        <div class="metric-card {accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(
    title: str,
    status: str,
    detail: str,
    online: bool = True,
) -> None:
    state_class = "online" if online else "offline"
    dot = "●"
    st.markdown(
        f"""
        <div class="status-card {state_class}">
            <div class="status-top">
                <span class="status-dot">{dot}</span>
                <span class="status-title">{title}</span>
                <span class="status-pill">{status}</span>
            </div>
            <div class="status-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_callout(
    title: str,
    body: str,
    kind: str = "info",
) -> None:
    st.markdown(
        f"""
        <div class="callout {kind}">
            <div class="callout-title">{title}</div>
            <div class="callout-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_card(
    prediction: int,
    probability: float,
    model_name: str,
    interpretation: str,
) -> None:
    percent = probability * 100
    state = "high" if probability >= 0.70 else "medium" if probability >= 0.40 else "low"
    state_label = {
        "high": "HIGH",
        "medium": "MODERATE",
        "low": "LOW",
    }[state]

    st.markdown(
        f"""
        <div class="prediction-card {state}">
            <div class="prediction-top">
                <div>
                    <div class="prediction-kicker">MODEL ANALYSIS</div>
                    <div class="prediction-title">{interpretation}</div>
                </div>
                <div class="prediction-badge">CLASS {prediction}</div>
            </div>
            <div class="probability-row">
                <div class="probability-value">{percent:.1f}%</div>
                <div class="probability-label">Predicted probability of job-change intention</div>
            </div>
            <div class="probability-track">
                <div class="probability-fill {state}" style="width:{percent:.1f}%"></div>
            </div>
            <div class="prediction-footer">
                <span>Model: <strong>{model_name}</strong></span>
                <span>Signal: <strong>{state_label}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_stat(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="mini-stat">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">◌</div>
            <div class="empty-title">{title}</div>
            <div class="empty-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_button(
    label: str,
    data: bytes | str,
    file_name: str,
    mime: str,
    key: str | None = None,
) -> None:
    st.download_button(
        label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
        use_container_width=False,
    )


def safe_style_metrics(df: pd.DataFrame) -> Any:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    styler = df.style
    if numeric_cols:
        styler = styler.format({col: "{:.4f}" for col in numeric_cols})
    return styler

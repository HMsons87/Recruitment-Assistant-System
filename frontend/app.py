from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import APIClient
from components import (
    empty_state,
    info_callout,
    inject_css,
    metric_card,
    mini_stat,
    page_header,
    prediction_card,
    section_header,
    status_card,
)


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

API_DEFAULT = os.getenv(
    "API_URL",
    "http://localhost:8000",
).rstrip("/")

METRICS_PATH = PROJECT_ROOT / "outputs" / "results" / "models_benchmark_metrics.csv"
FEATURES_PATH = PROJECT_ROOT / "outputs" / "results" / "top_predictive_features.csv"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_candidates.csv"
TEMPLATE_PATH = PROJECT_ROOT / "data" / "sample_candidates.csv"


# -----------------------------------------------------------------------------
# STYLING
# -----------------------------------------------------------------------------

CSS_PATH = Path(__file__).with_name("styles.css")
if CSS_PATH.exists():
    inject_css(CSS_PATH.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> APIClient:
    return APIClient(base_url)


@st.cache_data(ttl=60, show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def api_error_message(error: str | None) -> str:
    if not error:
        return "Unknown backend error."
    return error


def build_candidate_from_form(values: dict) -> dict:
    return {
        "city_development_index": float(values["city_development_index"]),
        "relevent_experience": str(values["relevent_experience"]),
        "enrolled_university": str(values["enrolled_university"]),
        "education_level": str(values["education_level"]),
        "major_discipline": str(values["major_discipline"]),
        "experience": str(values["experience"]),
        "company_size": str(values["company_size"]),
        "company_type": str(values["company_type"]),
        "last_new_job": str(values["last_new_job"]),
        "training_hours": float(values["training_hours"]),
    }


def render_sidebar(
    active_page: str,
    health_data: dict | None,
    api_url: str,
) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.3rem 0 1.1rem;">
                <div style="font-size:1.35rem;font-weight:900;color:#eef4ff;">🧠 SRA</div>
                <div style="color:#91a2ba;font-size:0.72rem;margin-top:0.18rem;">
                    Smart Recruitment Assistant
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='color:#6f839e;font-size:0.67rem;font-weight:800;letter-spacing:.12em;margin-bottom:.35rem;'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        pages = [
            "Dashboard",
            "Candidate Screening",
            "Batch Analysis",
            "Model Insights",
            "System",
        ]

        selected = st.radio(
            "Navigation",
            pages,
            index=pages.index(active_page),
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown(
            "<div style='color:#6f839e;font-size:0.67rem;font-weight:800;letter-spacing:.12em;margin-bottom:.45rem;'>SYSTEM</div>",
            unsafe_allow_html=True,
        )

        if health_data:
            status_card(
                "FastAPI Backend",
                "ONLINE" if health_data.get("model_loaded") else "MODEL NOT READY",
                f"Model: {health_data.get('model', 'Unknown')}",
                online=bool(health_data.get("model_loaded", False)),
            )
        else:
            status_card(
                "FastAPI Backend",
                "OFFLINE",
                "Start the backend on port 8000.",
                online=False,
            )

        st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='color:#6f839e;font-size:0.67rem;font-weight:800;letter-spacing:.12em;margin-bottom:.35rem;'>CONNECTION</div>",
            unsafe_allow_html=True,
        )

        with st.form("api_connection_form"):
            api_url_input = st.text_input(
                "FastAPI URL",
                value=api_url,
                help="Local default: http://localhost:8000",
            ).strip().rstrip("/")
            apply_api_url = st.form_submit_button(
                "Apply API URL",
                use_container_width=True,
            )

        if apply_api_url:
            st.session_state["api_url"] = api_url_input or "http://localhost:8000"
            st.rerun()

        st.caption("Decision-support prototype • v1.0")

    return selected


# -----------------------------------------------------------------------------
# API INITIALIZATION
# -----------------------------------------------------------------------------

if "api_url" not in st.session_state:
    st.session_state["api_url"] = API_DEFAULT

api = get_api_client(st.session_state["api_url"])
health_result = api.health()
health_data = health_result.data if health_result.ok else None

active_page = st.session_state.get("active_page", "Dashboard")
active_page = render_sidebar(active_page, health_data, api.base_url)
st.session_state.active_page = active_page


# -----------------------------------------------------------------------------
# GLOBAL NOTICE
# -----------------------------------------------------------------------------

info_callout(
    "Target definition",
    "The model predicts job-change intention. Class 1 means the candidate is looking for a job change; class 0 means the candidate is not. This is not a hiring-quality score.",
    kind="info",
)


# -----------------------------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------------------------

if active_page == "Dashboard":
    page_header(
        "Executive Overview",
        "Smart Recruitment Assistant",
        "A clean operational view of your candidate data, model health, and predictive signals.",
        badge="ML + API + BI",
    )

    model_info_result = api.model_info()
    model_info = model_info_result.data if model_info_result.ok else None

    df = load_csv(str(DATA_PATH))
    metrics = load_csv(str(METRICS_PATH))
    features = load_csv(str(FEATURES_PATH))

    if df is not None:
        total_candidates = len(df)
        target_rate = float(df["target"].mean()) if "target" in df.columns else 0.0
        avg_training = float(df["training_hours"].mean()) if "training_hours" in df.columns else 0.0
        avg_cdi = float(df["city_development_index"].mean()) if "city_development_index" in df.columns else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Candidates", f"{total_candidates:,}", "Processed candidate profiles", "violet")
        with c2:
            metric_card("Target = 1", f"{target_rate:.1%}", "Looking for a job change", "cyan")
        with c3:
            metric_card("Avg. Training", f"{avg_training:.1f} h", "Across the candidate population", "violet")
        with c4:
            metric_card("Avg. City Index", f"{avg_cdi:.3f}", "Dataset-level context", "cyan")
    else:
        empty_state(
            "Processed dataset not found",
            "Run the Notebook first and make sure data/processed/cleaned_candidates.csv exists.",
        )

    st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 0.95], gap="large")

    with col_left:
        section_header("Candidate population", "The core distribution the model sees.")

        if df is not None and "target" in df.columns:
            counts = df["target"].value_counts().sort_index()
            chart_df = pd.DataFrame(
                {
                    "Target": [
                        "Not looking for job change",
                        "Looking for job change",
                    ],
                    "Count": [
                        int(counts.get(0, 0)),
                        int(counts.get(1, 0)),
                    ],
                }
            )

            fig = px.pie(
                chart_df,
                names="Target",
                values="Count",
                hole=0.62,
                template="plotly_dark",
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate="%{label}<br>%{value:,} candidates<extra></extra>",
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.05),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            empty_state("No candidate data", "The processed dataset is required for the overview charts.")

    with col_right:
        section_header("Model performance", "The primary model is selected using F1, with ROC-AUC as a secondary signal.")

        if metrics is not None and not metrics.empty:
            metric_display = metrics.copy()
            # التأكد من وجود عمود Model بشكل صحيح ودون تكرار
            if "Model" not in metric_display.columns:
                metric_display = metric_display.reset_index()
                if "index" in metric_display.columns:
                    metric_display = metric_display.rename(columns={"index": "Model"})
                elif metric_display.columns[0] != "Model":
                    metric_display = metric_display.rename(columns={metric_display.columns[0]: "Model"})

            numeric_cols = [
                col for col in metric_display.columns
                if col != "Model" and pd.api.types.is_numeric_dtype(metric_display[col])
            ]

            st.dataframe(
                metric_display.style.format(
                    {col: "{:.4f}" for col in numeric_cols}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            empty_state("Metrics not available", "Run the Notebook to generate model_benchmark metrics.")

        if model_info:
            mini_stat("Primary model", model_info.get("model", "Unknown"))
            mini_stat("Selection metric", model_info.get("selection_metric", "F1 Score"))

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        section_header("Top predictive features", "Model-derived importance — not causal influence.")
        if features is not None and not features.empty:
            top_features = features.head(10).sort_values("Importance")
            fig = px.bar(
                top_features,
                x="Importance",
                y="Feature",
                orientation="h",
                template="plotly_dark",
            )
            fig.update_layout(
                margin=dict(l=10, r=20, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=390,
                xaxis_title="Importance",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            empty_state("Feature importance not found", "Run the Notebook and generate top_predictive_features.csv.")

    with col_b:
        section_header("Quick actions", "Move directly into an operational workflow.")
        st.markdown(
            """
            <div class="callout info">
                <div class="callout-title">Single candidate screening</div>
                <div class="callout-body">Analyze one profile through the FastAPI inference endpoint and inspect the predicted probability.</div>
            </div>
            <div class="callout info">
                <div class="callout-title">Batch analysis</div>
                <div class="callout-body">Upload a CSV, send the profiles to the backend, and return a ranked result table.</div>
            </div>
            <div class="callout info">
                <div class="callout-title">Model insights</div>
                <div class="callout-body">Inspect performance metrics and model-derived predictive features.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if TEMPLATE_PATH.exists():
            st.download_button(
                "Download CSV template",
                data=TEMPLATE_PATH.read_bytes(),
                file_name="sample_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# SINGLE CANDIDATE SCREENING
# -----------------------------------------------------------------------------

elif active_page == "Candidate Screening":
    page_header(
        "Candidate Analysis",
        "Screen a Candidate",
        "Enter a candidate profile and receive an inference from the trained ML pipeline through FastAPI.",
        badge="POST /predict",
    )

    with st.form("candidate_form", clear_on_submit=False):
        st.markdown("### Candidate profile")
        st.caption("Fields are validated by the FastAPI backend before inference.")

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**Education & experience**")
            education_level = st.selectbox(
                "Education level",
                ["Primary School", "High School", "Graduate", "Masters", "Phd"],
            )
            enrolled_university = st.selectbox(
                "University enrollment",
                ["no_enrollment", "Full time course", "Part time course"],
            )
            major_discipline = st.selectbox(
                "Major discipline",
                ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"],
            )
            experience = st.selectbox(
                "Experience",
                ["<1"] + [str(i) for i in range(1, 21)] + [">20"],
            )
            relevent_experience = st.selectbox(
                "Relevant experience",
                ["Has relevent experience", "No relevent experience"],
            )

        with right:
            st.markdown("**Professional context**")
            company_size = st.selectbox(
                "Company size",
                [
                    "<10", "10/49", "50-99", "100-500", "500-999",
                    "1000-4999", "5000-9999", "10000+",
                ],
            )
            company_type = st.selectbox(
                "Company type",
                [
                    "Pvt Ltd", "Funded Startup", "Public Sector",
                    "NGO", "Early Stage Startup", "Other",
                ],
            )
            last_new_job = st.selectbox(
                "Last new job",
                ["never", "1", "2", "3", "4", ">4"],
            )
            city_development_index = st.number_input(
                "City development index",
                min_value=0.0,
                max_value=1.0,
                value=0.80,
                step=0.001,
                format="%.3f",
            )
            training_hours = st.number_input(
                "Training hours",
                min_value=0.0,
                max_value=500.0,
                value=40.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "Analyze candidate",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        candidate = build_candidate_from_form(locals())
        with st.spinner("Sending profile to FastAPI and running inference…"):
            result = api.predict(candidate)

        if not result.ok:
            st.error(f"Prediction failed: {api_error_message(result.error)}")
        else:
            payload = result.data
            st.session_state["last_prediction"] = payload

    payload = st.session_state.get("last_prediction")
    if payload:
        section_header("Analysis result", "This result is a model prediction, not an autonomous hiring decision.")
        prediction_card(
            payload["prediction"],
            payload["job_change_probability"],
            payload["model"],
            payload["interpretation"],
        )

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Predicted class", str(payload["prediction"]), "0 = not looking, 1 = looking", "violet")
        with c2:
            metric_card("Probability", f"{payload['job_change_percentage']:.2f}%", "Predicted probability of class 1", "cyan")
        with c3:
            metric_card("Model", payload["model"], "Loaded from saved pipeline", "violet")


# -----------------------------------------------------------------------------
# BATCH ANALYSIS
# -----------------------------------------------------------------------------

elif active_page == "Batch Analysis":
    page_header(
        "Batch Operations",
        "Analyze Candidate Pool",
        "Upload a candidate CSV, send it to the backend, and review the ranked output without putting ML logic in the frontend.",
        badge="POST /predict/batch",
    )

    required_columns = [
        "city_development_index",
        "relevent_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "experience",
        "company_size",
        "company_type",
        "last_new_job",
        "training_hours",
    ]

    top_left, top_right = st.columns([1.4, 0.6])

    with top_left:
        uploaded_file = st.file_uploader(
            "Upload candidate CSV",
            type=["csv"],
            help="Use the sample template for the exact feature names.",
        )

    with top_right:
        if TEMPLATE_PATH.exists():
            st.download_button(
                "Download template",
                data=TEMPLATE_PATH.read_bytes(),
                file_name="sample_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if uploaded_file is None:
        empty_state(
            "Ready for a candidate file",
            "Upload a CSV with the required model features. An optional enrollee_id column will be preserved for reporting.",
        )
    else:
        batch_df = pd.read_csv(uploaded_file)
        missing = [col for col in required_columns if col not in batch_df.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif len(batch_df) > 10000:
            st.error("Maximum supported batch size is 10,000 candidates.")
        else:
            section_header("Input preview", f"{len(batch_df):,} candidate records loaded.")
            st.dataframe(batch_df.head(12), use_container_width=True, hide_index=True)

            analyze = st.button(
                "Analyze candidate pool",
                type="primary",
                use_container_width=True,
            )

            if analyze:
                # تجهيز البيانات وتوحيد أنواع النصوص وتفريغ NaNs لضمان توافق Pydantic
                batch_df_cleaned = batch_df.copy()
                str_cols = [
                    "experience",
                    "last_new_job",
                    "company_size",
                    "company_type",
                    "education_level",
                    "enrolled_university",
                    "major_discipline",
                    "relevent_experience",
                ]
                for col in str_cols:
                    if col in batch_df_cleaned.columns:
                        batch_df_cleaned[col] = batch_df_cleaned[col].apply(
                            lambda v: str(int(v))
                            if isinstance(v, (int, float)) and pd.notna(v) and float(v).is_integer()
                            else (str(v).strip() if pd.notna(v) else None)
                        )

                for col in ["city_development_index", "training_hours"]:
                    if col in batch_df_cleaned.columns:
                        batch_df_cleaned[col] = pd.to_numeric(batch_df_cleaned[col], errors="coerce")

                candidates = batch_df_cleaned.where(pd.notna(batch_df_cleaned), None).to_dict(orient="records")

                with st.spinner(f"Analyzing {len(candidates):,} candidates…"):
                    result = api.predict_batch(candidates)

                if not result.ok:
                    st.error(f"Batch prediction failed: {api_error_message(result.error)}")
                else:
                    items = result.data["results"]
                    rows: list[dict] = []
                    for item in items:
                        row = item["candidate"].copy()
                        row.update(
                            {
                                "Rank": item["rank"],
                                "Predicted_Class": item["prediction"],
                                "Predicted_Job_Change_Probability": item["job_change_probability"],
                                "Predicted_Job_Change_Percentage": item["job_change_percentage"],
                                "Interpretation": item["interpretation"],
                            }
                        )
                        rows.append(row)

                    ranked = pd.DataFrame(rows)
                    st.session_state["ranked_results"] = ranked
                    st.session_state["ranked_model"] = result.data.get("model", "Unknown")

    ranked = st.session_state.get("ranked_results")
    ranked_model = st.session_state.get("ranked_model", "Unknown")

    if isinstance(ranked, pd.DataFrame) and not ranked.empty:
        section_header("Ranking result", f"Model: {ranked_model}")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Candidates analyzed", f"{len(ranked):,}", "Returned by the FastAPI batch endpoint", "violet")
        with c2:
            metric_card("Top candidate signal", f"{ranked.iloc[0]['Predicted_Job_Change_Percentage']:.1f}%", "Highest predicted class-1 probability", "cyan")
        with c3:
            metric_card("Model", ranked_model, "Current backend pipeline", "violet")

        top10 = ranked.head(10).copy()

        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
        st.markdown("**Top 10**")
        st.dataframe(top10, use_container_width=True, hide_index=True)

        st.markdown("**All ranked candidates**")
        st.dataframe(ranked, use_container_width=True, hide_index=True)

        st.download_button(
            "Download ranked results",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )

        info_callout(
            "Ranking interpretation",
            "The ranking is based on predicted job-change intention. It does not represent a true hiring-quality ranking because the dataset does not contain historical hiring or selection outcomes.",
            kind="warning",
        )


# -----------------------------------------------------------------------------
# MODEL INSIGHTS
# -----------------------------------------------------------------------------

elif active_page == "Model Insights":
    page_header(
        "Model Intelligence",
        "Performance & Explainability",
        "Explore the metrics that define model behavior and the features the Random Forest relied on most.",
        badge="BI + ML",
    )

    metrics = load_csv(str(METRICS_PATH))
    features = load_csv(str(FEATURES_PATH))

    if metrics is not None and not metrics.empty:
        section_header("Model comparison", "F1 is used as the primary selection metric because the target classes are imbalanced.")

        metric_long = metrics.copy()
        # التأكد من فك الـ index أو التعرف على عمود Model بدون إطلاق ValueError
        if "Model" not in metric_long.columns:
            metric_long = metric_long.reset_index()
            if "index" in metric_long.columns:
                metric_long = metric_long.rename(columns={"index": "Model"})
            elif metric_long.columns[0] != "Model":
                metric_long = metric_long.rename(columns={metric_long.columns[0]: "Model"})

        numeric_metric_cols = [
            c for c in metric_long.columns 
            if c != "Model" and pd.api.types.is_numeric_dtype(metric_long[c])
        ]

        fig = go.Figure()
        for metric_name in numeric_metric_cols:
            fig.add_trace(
                go.Bar(
                    name=metric_name,
                    x=metric_long["Model"],
                    y=metric_long[metric_name],
                )
            )
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(range=[0, 1], title="Score"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(
            metric_long.style.format(
                {col: "{:.4f}" for col in numeric_metric_cols}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        empty_state("Model metrics not available", "Run the Notebook to generate the metrics artifact.")

    if features is not None and not features.empty:
        section_header("Predictive feature importance", "These are model-derived signals and should not be interpreted as causal effects.")

        top_features = features.head(12).sort_values("Importance")
        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark",
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=20, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Importance",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        empty_state("Feature importance not available", "Run the Notebook to generate top_predictive_features.csv.")


# -----------------------------------------------------------------------------
# SYSTEM / ABOUT
# -----------------------------------------------------------------------------

else:
    page_header(
        "System",
        "Architecture & Runtime",
        "A compact view of how the frontend, API, model artifact, and deployment pieces fit together.",
        badge="SYSTEM",
    )

    info_col, status_col = st.columns([1.2, 0.8], gap="large")

    with info_col:
        section_header("Architecture")
        st.markdown(
            """
            <div class="callout info">
                <div class="callout-title">Frontend</div>
                <div class="callout-body">Streamlit handles navigation, forms, charts, tables, and user interaction only.</div>
            </div>
            <div class="callout info">
                <div class="callout-title">Backend</div>
                <div class="callout-body">FastAPI exposes /health, /model-info, /predict, and /predict/batch.</div>
            </div>
            <div class="callout info">
                <div class="callout-title">Model</div>
                <div class="callout-body">The backend loads the complete saved scikit-learn pipeline, keeping preprocessing and inference consistent.</div>
            </div>
            <div class="callout info">
                <div class="callout-title">Deployment</div>
                <div class="callout-body">Docker Compose can run the frontend and backend as separate services.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col:
        section_header("Runtime status")
        if health_data:
            status_card(
                "FastAPI Backend",
                health_data.get("status", "UNKNOWN").upper(),
                f"Model: {health_data.get('model', 'Unknown')}",
                online=bool(health_data.get("model_loaded", False)),
            )

            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

            info_result = api.model_info()
            if info_result.ok:
                info = info_result.data
                mini_stat("Primary model", info.get("model", "Unknown"))
                mini_stat("Selection metric", info.get("selection_metric", "F1 Score"))
                mini_stat("Excluded features", ", ".join(info.get("excluded_from_model", [])))
        else:
            status_card(
                "FastAPI Backend",
                "OFFLINE",
                api_error_message(health_result.error),
                online=False,
            )

    section_header("Operational commands")
    st.code(
        """# Terminal 1 — FastAPI\nuvicorn backend.main:app --reload --port 8000\n\n# Terminal 2 — Streamlit\nstreamlit run frontend/app.py\n\n# Or Docker\ndocker compose up --build""",
        language="bash",
    )

    info_callout(
        "Important",
        "The Notebook is the training and evaluation environment. The production-style runtime uses the saved pipeline through FastAPI; Streamlit does not retrain the model.",
        kind="warning",
    )
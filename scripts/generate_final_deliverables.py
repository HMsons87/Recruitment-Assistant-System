from __future__ import annotations

from pathlib import Path
import json
import math
import textwrap

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
REPORTS = ROOT / "reports"
PRESENTATION = ROOT / "presentation"

METRICS = RESULTS / "models_benchmark_metrics.csv"
FEATURES = RESULTS / "top_predictive_features.csv"
RANKING = RESULTS / "top_10_recommended_candidates.csv"
CLEANED = ROOT / "data" / "processed" / "cleaned_candidates.csv"

REPORTS.mkdir(parents=True, exist_ok=True)
PRESENTATION.mkdir(parents=True, exist_ok=True)


def load_metrics():
    if METRICS.exists():
        return pd.read_csv(METRICS, index_col=0)
    return None


def metric_text(metrics, model, col):
    if metrics is None or model not in metrics.index or col not in metrics.columns:
        return "TBD"
    return f"{float(metrics.loc[model, col]):.4f}"


def best_model(metrics):
    if metrics is None:
        return "TBD"
    ranked = metrics.sort_values(["F1 Score", "ROC-AUC"], ascending=False)
    return ranked.index[0]


def write_markdown():
    metrics = load_metrics()
    model_name = best_model(metrics)

    if FEATURES.exists():
        feat_df = pd.read_csv(FEATURES).head(10)
        feature_lines = "\n".join(
            f"- `{row.Feature}`: {row.Importance:.4f}" for row in feat_df.itertuples()
        )
    else:
        feature_lines = "- [Run the notebook to populate the feature-importance table.]"

    text = f"""# Smart Recruitment Assistant - Project Documentation

## 1. Problem Statement

Organizations may receive large numbers of candidate profiles. Manual screening can be time-consuming. This project develops an educational HR analytics prototype that uses structured candidate information to predict **job-change intention** and provide interpretable analytics.

## 2. Dataset

Dataset: **HR Analytics: Job Change of Data Scientists**.

Target definition:

- `0` = Not looking for a job change
- `1` = Looking for a job change

The dataset is imbalanced, so accuracy is not treated as the only evaluation criterion.

## 3. Data Preparation

The project removes duplicate rows, keeps candidate identifiers for traceability, and excludes `enrollee_id`, `city`, and `gender` from the baseline predictive model. Feature engineering creates `experience_years`, `training_hours_log`, and `training_intensity`.

Missing-value imputation, ordinal encoding, one-hot encoding, and scaling are learned inside scikit-learn pipelines **after the train/test split** to avoid data leakage.

## 4. Machine Learning Models

Two required classifiers are trained:

1. Logistic Regression with `class_weight='balanced'`.
2. Random Forest with `n_estimators=200`, `max_depth=10`, and `class_weight='balanced'`.

## 5. Model Evaluation

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | {metric_text(metrics, 'Logistic Regression', 'Accuracy')} | {metric_text(metrics, 'Logistic Regression', 'Precision')} | {metric_text(metrics, 'Logistic Regression', 'Recall')} | {metric_text(metrics, 'Logistic Regression', 'F1 Score')} | {metric_text(metrics, 'Logistic Regression', 'ROC-AUC')} |
| Random Forest | {metric_text(metrics, 'Random Forest', 'Accuracy')} | {metric_text(metrics, 'Random Forest', 'Precision')} | {metric_text(metrics, 'Random Forest', 'Recall')} | {metric_text(metrics, 'Random Forest', 'F1 Score')} | {metric_text(metrics, 'Random Forest', 'ROC-AUC')} |

**Primary model selected using F1 Score:** {model_name}

## 6. Business Intelligence Insights

Top predictive features from the Random Forest model:

{feature_lines}

These values represent predictive importance, not causal effects. A feature with high importance does not by itself prove that it causes job-change behavior.

## 7. Bonus Ranking

The optional ranking module orders candidates by predicted **job-change intention probability**. It must not be described as a ranking of the “best applicants,” because the primary dataset does not contain a historical hiring/selection target.

## 8. Dashboard

The Streamlit application provides:

- Overview statistics
- Model performance table
- Top predictive features
- Single-candidate prediction
- Batch candidate ranking
- CSV export of ranked predictions

## 9. Limitations

The main limitation is the target definition: it measures job-change intention rather than a real hiring or selection outcome. Therefore, the system should be treated as an HR analytics / decision-support prototype and not as an autonomous hiring system.

## 10. Conclusion

The project demonstrates a complete tabular machine-learning workflow from data preparation and exploratory analysis to model comparison, interpretation, artifact saving, and interactive application deployment.

> **Refresh rule:** if any result above shows `[RUN NOTEBOOK]`, run `notebooks/smart_recruitment_assistant.ipynb` first, then run `python scripts/generate_final_deliverables.py` again.
"""
    path = REPORTS / "Documentation.md"
    path.write_text(text, encoding="utf-8")
    return path


def make_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors

    metrics = load_metrics()
    model_name = best_model(metrics)
    pdf_path = REPORTS / "Documentation.pdf"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24, spaceAfter=18))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))

    story = [
        Paragraph("Smart Recruitment Assistant", styles["TitleCenter"]),
        Paragraph("ITI AI Level 1 - Project Documentation", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph("Target: 1 = Looking for a job change; 0 = Not looking for a job change.", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("1. Problem Statement", styles["Heading2"]),
        Paragraph("The project builds an educational HR analytics prototype that analyzes structured candidate profiles and predicts job-change intention, while providing model evaluation and business insights.", styles["BodyText"]),
        Paragraph("2. Data Preparation", styles["Heading2"]),
        Paragraph("Duplicate rows are removed. Candidate IDs are retained for traceability but are not used as predictive features. The baseline excludes city and gender from the predictive model. Feature engineering creates experience_years, training_hours_log, and training_intensity. Missing values and encoders are learned inside pipelines after the train/test split.", styles["BodyText"]),
        Paragraph("3. Models", styles["Heading2"]),
        Paragraph("Logistic Regression and Random Forest are trained with class balancing. F1 Score is the primary model-selection metric because the target classes are imbalanced; ROC-AUC is used as a secondary metric.", styles["BodyText"]),
        Paragraph("4. Evaluation", styles["Heading2"]),
    ]

    data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        ["Logistic Regression", metric_text(metrics, "Logistic Regression", "Accuracy"), metric_text(metrics, "Logistic Regression", "Precision"), metric_text(metrics, "Logistic Regression", "Recall"), metric_text(metrics, "Logistic Regression", "F1 Score"), metric_text(metrics, "Logistic Regression", "ROC-AUC")],
        ["Random Forest", metric_text(metrics, "Random Forest", "Accuracy"), metric_text(metrics, "Random Forest", "Precision"), metric_text(metrics, "Random Forest", "Recall"), metric_text(metrics, "Random Forest", "F1 Score"), metric_text(metrics, "Random Forest", "ROC-AUC")],
    ]
    table = Table(data, repeatRows=1, colWidths=[110, 60, 60, 60, 50, 60])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 10), Paragraph(f"Primary model by F1 Score: {model_name}", styles["BodyText"])]

    story += [
        PageBreak(),
        Paragraph("5. Business Intelligence", styles["Heading2"]),
        Paragraph("Random Forest feature importance is reported as predictive importance, not causal influence.", styles["BodyText"]),
        Spacer(1, 8),
    ]

    if FEATURES.exists():
        feat = pd.read_csv(FEATURES).head(10)
        feat_data = [["Feature", "Importance"]] + [[str(r.Feature), f"{r.Importance:.4f}"] for r in feat.itertuples()]
        ft = Table(feat_data, colWidths=[280, 100], repeatRows=1)
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story.append(ft)
    else:
        story.append(Paragraph("Feature-importance results are populated after the notebook is executed.", styles["BodyText"]))

    story += [
        Spacer(1, 14),
        Paragraph("6. Bonus Ranking", styles["Heading2"]),
        Paragraph("The optional ranking orders candidates by predicted job-change intention. It is not a true hiring ranking because the primary dataset does not include historical hiring/selection outcomes.", styles["BodyText"]),
        Paragraph("7. Dashboard", styles["Heading2"]),
        Paragraph("The Streamlit application provides overview statistics, model metrics, predictive features, single-candidate inference, batch ranking, and CSV export.", styles["BodyText"]),
        Paragraph("8. Limitations", styles["Heading2"]),
        Paragraph("The main limitation is the target definition. The model predicts job-change intention rather than an actual hiring outcome, so it should be treated as an HR analytics decision-support prototype.", styles["BodyText"]),
        Spacer(1, 18),
        Paragraph("Refresh this document after running the notebook by executing: python scripts/generate_final_deliverables.py", styles["Small"]),
    ]

    doc.build(story)
    return pdf_path


def make_pptx():
    from pptx import Presentation
    from pptx.util import Inches, Pt

    metrics = load_metrics()
    model_name = best_model(metrics)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def add_title_slide(title, subtitle):
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = title
        slide.placeholders[1].text = subtitle
        return slide

    def add_bullets(title, bullets):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        tf = slide.placeholders[1].text_frame
        tf.clear()
        for i, bullet in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = bullet
            p.level = 0
            p.font.size = Pt(20)
        return slide

    add_title_slide("Smart Recruitment Assistant", "ITI AI Level 1 - Graduation Project")
    add_bullets("Business Problem", [
        "Large candidate volumes make manual screening time-consuming.",
        "The system provides data-driven HR analytics and prediction.",
        "The output is decision support, not autonomous hiring.",
    ])
    add_bullets("Dataset & Target", [
        "HR Analytics: Job Change of Data Scientists.",
        "target=0: not looking for a job change.",
        "target=1: looking for a job change.",
        "The target is not a historical hire/reject label.",
    ])
    add_bullets("Data Preparation", [
        "Duplicate removal and missing-value analysis.",
        "Feature engineering: experience_years, training_hours_log, training_intensity.",
        "Preprocessing is learned inside pipelines after the train/test split.",
    ])
    add_bullets("Machine Learning", [
        "Logistic Regression with class balancing.",
        "Random Forest with 200 trees, max depth 10, and class balancing.",
        "Both models use the same leakage-safe preprocessing design.",
    ])

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Model Evaluation"
    rows, cols = 3, 6
    table = slide.shapes.add_table(rows, cols, Inches(0.6), Inches(1.6), Inches(12.1), Inches(2.2)).table
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for c, h in enumerate(headers):
        table.cell(0, c).text = h
    for r, model in enumerate(["Logistic Regression", "Random Forest"], start=1):
        table.cell(r, 0).text = model
        table.cell(r, 1).text = metric_text(metrics, model, "Accuracy")
        table.cell(r, 2).text = metric_text(metrics, model, "Precision")
        table.cell(r, 3).text = metric_text(metrics, model, "Recall")
        table.cell(r, 4).text = metric_text(metrics, model, "F1 Score")
        table.cell(r, 5).text = metric_text(metrics, model, "ROC-AUC")
    box = slide.shapes.add_textbox(Inches(0.9), Inches(4.2), Inches(11.5), Inches(1.0))
    box.text_frame.text = f"Primary model selected by F1 Score: {model_name}"
    box.text_frame.paragraphs[0].font.size = Pt(24)

    if FEATURES.exists():
        feats = pd.read_csv(FEATURES).head(10)
        bullets = [f"{r.Feature}: {r.Importance:.4f}" for r in feats.itertuples()]
    else:
        bullets = ["Feature importance is populated after notebook execution."]
    add_bullets("BI Insights", bullets)
    add_bullets("Dashboard & Bonus", [
        "Streamlit overview dashboard.",
        "Single-candidate prediction.",
        "Batch candidate ranking and CSV export.",
        "Ranking is based on predicted job-change intention, not hiring quality.",
    ])
    add_bullets("Conclusion & Limitations", [
        "The project demonstrates a complete tabular ML workflow.",
        "Model outputs are predictive associations, not causal claims.",
        "A true hiring model requires historical hiring/selection labels.",
    ])

    path = PRESENTATION / "presentation.pptx"
    prs.save(path)
    return path


def main():
    md_path = write_markdown()
    pdf_path = make_pdf()
    pptx_path = make_pptx()
    print("Generated:")
    print(md_path)
    print(pdf_path)
    print(pptx_path)


if __name__ == "__main__":
    main()

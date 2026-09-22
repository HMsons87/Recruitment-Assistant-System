# Smart Recruitment Assistant

Educational HR analytics / machine-learning prototype for the ITI AI Level 1 graduation project.

## Important target definition

The primary dataset is **HR Analytics: Job Change of Data Scientists**. Its target is:

- `0` = Not looking for a job change
- `1` = Looking for a job change

Therefore, the model predicts **job-change intention**. It is not a real hiring/acceptance probability model.

## Architecture

```text
Raw CSV
   |
   v
Data inspection + EDA
   |
   v
Train/Test Split (stratified)
   |
   v
Feature Engineering
   |
   v
ColumnTransformer
   |---- Numeric: Median Imputation -> Scaling (Logistic Regression)
   |---- Ordinal: Most-Frequent Imputation -> Ordinal Encoding
   |---- Nominal: Most-Frequent Imputation -> One-Hot Encoding
   |
   +-------------------+
   |                   |
   v                   v
Logistic Regression   Random Forest
   |                   |
   +---------+---------+
             v
Evaluation: Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix / ROC Curve
             |
             v
Feature importance + BI insights
             |
             v
Saved pipeline + Streamlit dashboard + bonus ranking
```

## Project structure

```text
Smart-Recruitment-Assistant/
├── data/
│   ├── raw/aug_train.csv
│   ├── processed/
│   └── sample_candidates.csv
├── notebooks/smart_recruitment_assistant.ipynb
├── src/
│   ├── preprocessing.py
│   └── modeling.py
├── models/
├── outputs/
│   ├── figures/
│   └── results/
├── app/app.py
├── reports/
├── presentation/
├── scripts/
├── requirements.txt
└── README.md
```

## How to run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Put the dataset here

```text
data/raw/aug_train.csv
```

### 3. Run the notebook

Run `notebooks/smart_recruitment_assistant.ipynb` from top to bottom.

The notebook creates:

- `data/processed/cleaned_candidates.csv`
- `data/processed/transformed_features.csv`
- `models/recruitment_logistic_regression.pkl`
- `models/recruitment_random_forest.pkl`
- `models/best_recruitment_pipeline.pkl`
- `models/preprocessor_pipeline.pkl`
- `outputs/results/models_benchmark_metrics.csv`
- `outputs/results/top_predictive_features.csv`
- `outputs/results/top_10_recommended_candidates.csv`
- figures under `outputs/figures/`

### 4. Generate final report + presentation

```bash
python scripts/generate_final_deliverables.py
```

This reads the actual results produced by the notebook and creates the documentation and presentation using those results. If the results files do not yet exist, it creates fill-ready documents with explicit placeholders instead of inventing metrics.

### 5. Start Streamlit

From the project root:

```bash
streamlit run app/app.py
```

## Ranking note

The optional ranking module sorts candidates by predicted **job-change intention**. It must not be described as ranking the “best applicants” because the primary dataset does not contain a historical hiring/selection target.

A genuine hiring-ranking system would require data containing historical hiring or selection outcomes.

## Ethical / business note

This is an educational decision-support prototype. Model outputs should not be used as an autonomous hiring decision. Sensitive attributes such as gender are excluded from the baseline predictive model and should be considered separately for audit/fairness analysis.

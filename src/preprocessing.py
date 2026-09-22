from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features used by the trained ML pipelines."""
    data = data.copy()

    def parse_experience(value):
        if pd.isna(value):
            return np.nan
        value = str(value).strip()
        if value == "<1":
            return 0.0
        if value == ">20":
            return 21.0
        return pd.to_numeric(value, errors="coerce")

    def parse_last_new_job(value):
        if pd.isna(value):
            return np.nan
        job_map = {"never": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, ">4": 5.0}
        return job_map.get(str(value).strip(), np.nan)

    data["experience_years"] = data["experience"].apply(parse_experience)
    last_job_num = data["last_new_job"].apply(parse_last_new_job)

    data["training_hours_log"] = np.log1p(data["training_hours"])
    data["training_intensity"] = (
        data["training_hours"] / (data["experience_years"] + 1.0)
    )

    data["is_company_missing"] = data["company_size"].isnull().astype(float)
    data["is_company_type_missing"] = data["company_type"].isnull().astype(float)
    # 5. تفاعل مؤشر تنمية المدينة مع الخبرة
    data["cdi_x_experience"] = data["city_development_index"] * (data["experience_years"] + 1.0)

    data["experience_job_ratio"] = (data["experience_years"] + 1.0) / (last_job_num + 1.0)

    data["is_low_cdi"] = (data["city_development_index"] < 0.65).astype(float)

    return data


MODEL_EXCLUDED_COLUMNS = ["enrollee_id", "city", "gender"]
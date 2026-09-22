from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city_development_index: float = Field(ge=0.0, le=1.0)
    relevent_experience: str | None = None
    enrolled_university: str | None = None
    education_level: str | None = None
    major_discipline: str | None = None
    experience: str | None = None
    company_size: str | None = None
    company_type: str | None = None
    last_new_job: str | None = None
    training_hours: float = Field(ge=0.0)


class PredictionResponse(BaseModel):
    prediction: int
    job_change_probability: float
    job_change_percentage: float
    interpretation: str
    model: str


class BatchPredictionRequest(BaseModel):
    candidates: list[CandidateProfile] = Field(min_length=1, max_length=10000)


class BatchCandidateResult(BaseModel):
    rank: int
    prediction: int
    job_change_probability: float
    job_change_percentage: float
    interpretation: str
    candidate: dict


class BatchPredictionResponse(BaseModel):
    count: int
    model: str
    results: list[BatchCandidateResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model: str
    selection_metric: str
    target_definition: dict[str, str]
    excluded_from_model: list[str]

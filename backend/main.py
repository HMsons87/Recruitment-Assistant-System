from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    BatchCandidateResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CandidateProfile,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from backend.services.prediction_service import (
    ModelNotReadyError,
    prediction_service,
)
from backend.services.ranking_service import ranking_service


app = FastAPI(
    title="Smart Recruitment Assistant API",
    description=(
        "ML inference API for job-change intention prediction. "
        "Educational decision-support prototype."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_model_ready() -> None:
    if not prediction_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not available. Run the notebook first so "
                "models/best_recruitment_pipeline.pkl is created."
            ),
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if prediction_service.is_ready else "model_not_ready",
        model=prediction_service.model_name,
        model_loaded=prediction_service.is_ready,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    ensure_model_ready()
    metadata = prediction_service.metadata
    return ModelInfoResponse(
        model=prediction_service.model_name,
        selection_metric=metadata.get("selection_metric", "F1 Score"),
        target_definition=metadata.get(
            "target_definition",
            {
                "0": "Not looking for a job change",
                "1": "Looking for a job change",
            },
        ),
        excluded_from_model=metadata.get(
            "excluded_from_model",
            ["enrollee_id", "city", "gender"],
        ),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(candidate: CandidateProfile) -> PredictionResponse:
    ensure_model_ready()

    try:
        prediction, probability = prediction_service.predict(
            candidate.model_dump()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        job_change_probability=round(probability, 6),
        job_change_percentage=round(probability * 100, 2),
        interpretation=prediction_service.interpretation(probability),
        model=prediction_service.model_name,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    ensure_model_ready()

    try:
        candidates = [candidate.model_dump() for candidate in request.candidates]
        ranked = ranking_service.rank(candidates)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    return BatchPredictionResponse(
        count=len(ranked),
        model=prediction_service.model_name,
        results=[BatchCandidateResult(**item) for item in ranked],
    )

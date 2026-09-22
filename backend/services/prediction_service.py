from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd

from src.preprocessing import MODEL_EXCLUDED_COLUMNS, add_engineered_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_recruitment_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


class ModelNotReadyError(RuntimeError):
    """Raised when the trained artifact does not exist."""


class PredictionService:
    """Loads the persisted ML pipeline once and performs inference."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pipeline = None
        self._metadata: dict = {}
        self.reload()

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        return self._metadata.get("best_model", "Unknown")

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    def reload(self) -> None:
        with self._lock:
            if not BEST_MODEL_PATH.exists():
                self._pipeline = None
                self._metadata = {}
                return

            self._pipeline = joblib.load(BEST_MODEL_PATH)

            if METADATA_PATH.exists():
                self._metadata = json.loads(
                    METADATA_PATH.read_text(encoding="utf-8")
                )
            else:
                self._metadata = {
                    "best_model": "Unknown",
                    "target_definition": {
                        "0": "Not looking for a job change",
                        "1": "Looking for a job change",
                    },
                    "excluded_from_model": MODEL_EXCLUDED_COLUMNS,
                }

    def _ensure_ready(self) -> None:
        if self._pipeline is None:
            raise ModelNotReadyError(
                "The trained model is not available. Run the notebook first."
            )

    @staticmethod
    def _prepare_dataframe(candidates: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(candidates)
        df = add_engineered_features(df)
        return df.drop(
            columns=MODEL_EXCLUDED_COLUMNS,
            errors="ignore",
        )

    def predict(self, candidate: dict) -> tuple[int, float]:
        self._ensure_ready()
        prepared = self._prepare_dataframe([candidate])
        prediction = int(self._pipeline.predict(prepared)[0])
        probability = float(self._pipeline.predict_proba(prepared)[0, 1])
        return prediction, probability

    def predict_batch(self, candidates: list[dict]) -> list[tuple[int, float]]:
        self._ensure_ready()
        prepared = self._prepare_dataframe(candidates)
        predictions = self._pipeline.predict(prepared)
        probabilities = self._pipeline.predict_proba(prepared)[:, 1]
        return [
            (int(prediction), float(probability))
            for prediction, probability in zip(predictions, probabilities)
        ]

    @staticmethod
    def interpretation(probability: float) -> str:
        if probability >= 0.70:
            return "High predicted job-change intention"
        if probability >= 0.40:
            return "Moderate predicted job-change intention"
        return "Low predicted job-change intention"


prediction_service = PredictionService()

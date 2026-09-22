from __future__ import annotations

from typing import Any

from .prediction_service import PredictionService, prediction_service


class RankingService:
    """Creates batch rankings using the backend's trained model."""

    def __init__(self, prediction_service: PredictionService) -> None:
        self.prediction_service = prediction_service

    def rank(self, candidates: list[dict]) -> list[dict[str, Any]]:
        predictions = self.prediction_service.predict_batch(candidates)

        ranked = []
        for original_index, (candidate, (prediction, probability)) in enumerate(
            zip(candidates, predictions)
        ):
            ranked.append(
                {
                    "_original_index": original_index,
                    "prediction": prediction,
                    "job_change_probability": probability,
                    "job_change_percentage": round(probability * 100, 2),
                    "interpretation": self.prediction_service.interpretation(
                        probability
                    ),
                    "candidate": candidate,
                }
            )

        ranked.sort(
            key=lambda item: item["job_change_probability"],
            reverse=True,
        )

        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
            item.pop("_original_index", None)

        return ranked


ranking_service = RankingService(prediction_service)

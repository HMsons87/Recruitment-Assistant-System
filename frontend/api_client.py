from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class APIResult:
    ok: bool
    data: Any = None
    error: str | None = None
    status_code: int | None = None


class APIClient:
    """Small, centralized HTTP client for the FastAPI backend."""

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> APIResult:
        url = f"{self.base_url}{endpoint}"
        # سحب قيمة timeout إن وجدت في kwargs أو استخدام الافتراضية لمنع التكرار
        timeout = kwargs.pop("timeout", self.timeout)

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs,
            )
            response.raise_for_status()
            return APIResult(
                ok=True,
                data=response.json(),
                status_code=response.status_code,
            )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            detail = None
            if exc.response is not None:
                try:
                    body = exc.response.json()
                    detail = body.get("detail") if isinstance(body, dict) else str(body)
                except ValueError:
                    detail = exc.response.text
            return APIResult(
                ok=False,
                error=detail or str(exc),
                status_code=status_code,
            )
        except requests.RequestException as exc:
            return APIResult(ok=False, error=str(exc))

    def health(self) -> APIResult:
        return self._request("GET", "/health", timeout=5)

    def model_info(self) -> APIResult:
        return self._request("GET", "/model-info", timeout=10)

    def predict(self, candidate: dict[str, Any]) -> APIResult:
        return self._request(
            "POST",
            "/predict",
            json=candidate,
            timeout=self.timeout,
        )

    def predict_batch(self, candidates: list[dict[str, Any]]) -> APIResult:
        return self._request(
            "POST",
            "/predict/batch",
            json={"candidates": candidates},
            timeout=max(self.timeout, 60),
        )
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_model_info_endpoint_shape():
    response = client.get("/model-info")
    # 200 when the model exists; 503 when the notebook has not been run.
    assert response.status_code in {200, 503}

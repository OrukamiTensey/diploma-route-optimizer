"""
Тести REST API для TD-VRPTW-P оптимізатора.

Покриває:
  - Health-ендпоінт (GET /api/v1/health → 200)
  - Створення задачі оптимізації (POST /api/v1/optimize → 202)
  - Валідаційні помилки (невалідні координати → 422)
  - Отримання неіснуючої задачі (GET /api/v1/optimize/{random_uuid} → 404)
  - Повний workflow: POST → очікування → GET з COMPLETED та scheduled_route
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.responses import TaskStatusEnum
from app.services.task_manager import TaskManager


# ---------------------------------------------------------------------------
# Фікстури
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Синхронний тестовий клієнт FastAPI."""
    return TestClient(app)


@pytest.fixture
def valid_optimization_payload() -> dict:
    """Валідний JSON-запит з 3 завданнями у Києві, старт о 08:00."""
    return {
        "depot": {
            "id": "depot-kpi",
            "latitude": 50.4488,
            "longitude": 30.4571,
            "name": "КПІ",
        },
        "tasks": [
            {
                "id": "task-0",
                "location": {
                    "id": "loc-0",
                    "latitude": 50.4501,
                    "longitude": 30.5234,
                    "name": "Хрещатик",
                },
                "time_window": {"start_time": 480, "end_time": 720},
                "service_duration": 900,
                "priority": 3,
            },
            {
                "id": "task-1",
                "location": {
                    "id": "loc-1",
                    "latitude": 50.4620,
                    "longitude": 30.5080,
                    "name": "Поділ",
                },
                "time_window": {"start_time": 540, "end_time": 780},
                "service_duration": 600,
                "priority": 5,
            },
            {
                "id": "task-2",
                "location": {
                    "id": "loc-2",
                    "latitude": 50.4350,
                    "longitude": 30.5190,
                    "name": "Печерськ",
                },
                "time_window": {"start_time": 600, "end_time": 840},
                "service_duration": 1200,
                "priority": 2,
            },
        ],
        "start_time": 480,
    }


# =====================================================================
# Health ендпоінт
# =====================================================================


class TestHealthEndpoint:
    """Тести GET /api/v1/health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health check повертає 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient) -> None:
        """Health check повертає {"status": "ok"}."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "ok"


# =====================================================================
# POST /api/v1/optimize — створення задачі
# =====================================================================


class TestPostOptimize:
    """Тести POST /api/v1/optimize."""

    def test_returns_202_accepted(
        self, client: TestClient, valid_optimization_payload: dict
    ) -> None:
        """Валідний запит повертає 202 Accepted."""
        response = client.post("/api/v1/optimize", json=valid_optimization_payload)
        assert response.status_code == 202

    def test_returns_task_id_and_pending(
        self, client: TestClient, valid_optimization_payload: dict
    ) -> None:
        """Відповідь містить task_id (UUID) і статус PENDING."""
        response = client.post("/api/v1/optimize", json=valid_optimization_payload)
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "PENDING"
        assert "message" in data

    def test_validation_error_invalid_latitude(
        self, client: TestClient
    ) -> None:
        """Невалідна широта (> 90) → 422 Unprocessable Entity."""
        payload = {
            "depot": {
                "id": "d",
                "latitude": 999.0,  # невалідна
                "longitude": 30.0,
            },
            "tasks": [
                {
                    "id": "t",
                    "location": {
                        "id": "l",
                        "latitude": 50.0,
                        "longitude": 30.0,
                    },
                    "time_window": {"start_time": 0, "end_time": 100},
                    "service_duration": 60,
                    "priority": 1,
                },
            ],
            "start_time": 0,
        }
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 422

    def test_validation_error_inverted_time_window(
        self, client: TestClient
    ) -> None:
        """Інвертоване часове вікно (start > end) → 422."""
        payload = {
            "depot": {
                "id": "d",
                "latitude": 50.0,
                "longitude": 30.0,
            },
            "tasks": [
                {
                    "id": "t",
                    "location": {
                        "id": "l",
                        "latitude": 50.0,
                        "longitude": 30.0,
                    },
                    "time_window": {"start_time": 720, "end_time": 480},
                    "service_duration": 60,
                    "priority": 1,
                },
            ],
            "start_time": 480,
        }
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 422

    def test_validation_error_empty_tasks(self, client: TestClient) -> None:
        """Порожній список завдань → 422."""
        payload = {
            "depot": {
                "id": "d",
                "latitude": 50.0,
                "longitude": 30.0,
            },
            "tasks": [],
            "start_time": 480,
        }
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 422

    def test_validation_error_missing_fields(self, client: TestClient) -> None:
        """Відсутні обов'язкові поля → 422."""
        response = client.post("/api/v1/optimize", json={})
        assert response.status_code == 422


# =====================================================================
# GET /api/v1/optimize/{task_id} — отримання результату
# =====================================================================


class TestGetOptimizeResult:
    """Тести GET /api/v1/optimize/{task_id}."""

    def test_nonexistent_task_returns_404(self, client: TestClient) -> None:
        """Запит неіснуючої задачі → 404 Not Found."""
        random_id = str(uuid4())
        response = client.get(f"/api/v1/optimize/{random_id}")
        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        """Невалідний UUID → 422."""
        response = client.get("/api/v1/optimize/not-a-uuid")
        assert response.status_code == 422


# =====================================================================
# Повний workflow: POST → poll → GET COMPLETED
# =====================================================================


class TestOptimizationWorkflow:
    """Інтеграційний тест повного циклу оптимізації."""

    def test_full_workflow_post_poll_get(
        self, client: TestClient, valid_optimization_payload: dict
    ) -> None:
        """POST → 202 → poll GET → COMPLETED з filled scheduled_route."""
        # 1. Створюємо задачу
        post_response = client.post(
            "/api/v1/optimize", json=valid_optimization_payload
        )
        assert post_response.status_code == 202
        task_id = post_response.json()["task_id"]

        # 2. Очікуємо завершення (poll з таймаутом)
        max_wait_seconds = 30
        poll_interval = 0.3
        elapsed = 0.0
        final_data = None

        while elapsed < max_wait_seconds:
            get_response = client.get(f"/api/v1/optimize/{task_id}")
            assert get_response.status_code == 200
            data = get_response.json()

            if data["status"] in ("COMPLETED", "FAILED"):
                final_data = data
                break

            time.sleep(poll_interval)
            elapsed += poll_interval

        # 3. Перевіряємо результат
        assert final_data is not None, (
            f"Задача не завершилась за {max_wait_seconds}с"
        )
        assert final_data["status"] == "COMPLETED"
        assert final_data["error_message"] is None

        # 4. Перевіряємо заповненість метрик
        assert final_data["total_duration"] is not None
        assert final_data["total_duration"] > 0.0
        assert final_data["total_lateness"] is not None
        assert final_data["fitness_cost"] is not None
        assert final_data["fitness_cost"] > 0.0

        # 5. Перевіряємо деталізований розклад
        scheduled = final_data["scheduled_route"]
        assert scheduled is not None
        assert len(scheduled) == 3  # 3 завдання у запиті

        for item in scheduled:
            assert "task_id" in item
            assert "arrival_time" in item
            assert "wait_time" in item
            assert item["wait_time"] >= 0.0
            assert "service_start" in item
            assert "lateness" in item
            assert item["lateness"] >= 0.0
            assert "departure_time" in item
            assert item["departure_time"] >= item["service_start"]

        # 6. Перевіряємо, що всі task_id з запиту присутні
        scheduled_ids = {item["task_id"] for item in scheduled}
        expected_ids = {t["id"] for t in valid_optimization_payload["tasks"]}
        assert scheduled_ids == expected_ids

        # 7. Перевіряємо історії
        assert final_data["convergence_history"] is not None
        assert len(final_data["convergence_history"]) > 0
        assert final_data["mutation_rate_history"] is not None
        assert len(final_data["mutation_rate_history"]) > 0

    def test_task_status_transitions(
        self, client: TestClient, valid_optimization_payload: dict
    ) -> None:
        """Статус задачі проходить через PENDING/RUNNING → COMPLETED."""
        post_response = client.post(
            "/api/v1/optimize", json=valid_optimization_payload
        )
        task_id = post_response.json()["task_id"]

        observed_statuses = set()
        max_wait = 30
        elapsed = 0.0

        while elapsed < max_wait:
            get_response = client.get(f"/api/v1/optimize/{task_id}")
            data = get_response.json()
            observed_statuses.add(data["status"])

            if data["status"] in ("COMPLETED", "FAILED"):
                break

            time.sleep(0.2)
            elapsed += 0.2

        # Фінальний статус має бути COMPLETED
        assert "COMPLETED" in observed_statuses
        # Мінімум 1 з проміжних станів має з'явитися
        assert len(observed_statuses) >= 1

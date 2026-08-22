"""
REST API ендпоінти v1 для оптимізації маршрутів TD-VRPTW-P.

Ендпоінти:
  POST /api/v1/optimize     — створити задачу оптимізації (HTTP 202 Accepted)
  GET  /api/v1/optimize/{id} — отримати статус/результат задачі
  GET  /api/v1/health        — перевірка працездатності сервісу
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.schemas.models import OptimizationRequest
from app.schemas.responses import (
    OptimizationResultResponse,
    OptimizationTaskCreateResponse,
)
from app.services.task_manager import task_manager

router = APIRouter()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    summary="Перевірка працездатності сервісу",
    tags=["system"],
)
async def health_check() -> dict:
    """Повертає статус працездатності API.

    Returns
    -------
    dict
        ``{"status": "ok"}``
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /optimize — створення задачі
# ---------------------------------------------------------------------------


@router.post(
    "/optimize",
    response_model=OptimizationTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запуск оптимізації маршруту",
    tags=["optimization"],
)
async def create_optimization_task(
    request: OptimizationRequest,
) -> OptimizationTaskCreateResponse:
    """Приймає запит на оптимізацію, валідує вхідні дані та ставить
    задачу до черги на фонове виконання.

    Parameters
    ----------
    request : OptimizationRequest
        Вхідні дані: депо, завдання, час старту, ваги.

    Returns
    -------
    OptimizationTaskCreateResponse
        Ідентифікатор задачі (UUID) та статус PENDING.
    """
    response = await task_manager.submit(request)
    return response


# ---------------------------------------------------------------------------
# GET /optimize/{task_id} — отримання результату
# ---------------------------------------------------------------------------


@router.get(
    "/optimize/{task_id}",
    response_model=OptimizationResultResponse,
    summary="Отримати статус та результат оптимізації",
    tags=["optimization"],
)
async def get_optimization_result(
    task_id: UUID,
) -> OptimizationResultResponse:
    """Повертає поточний статус та результат задачі оптимізації.

    Parameters
    ----------
    task_id : UUID
        Ідентифікатор задачі (отриманий з POST /optimize).

    Returns
    -------
    OptimizationResultResponse
        Статус, метрики маршруту, деталізований розклад.

    Raises
    ------
    HTTPException (404)
        Якщо задача з таким task_id не знайдена.
    """
    result = task_manager.get_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задачу з id={task_id} не знайдено",
        )
    return result

"""
Pydantic v2 моделі відповідей REST API для TD-VRPTW-P оптимізатора.

Визначає структуру відповідей для:
  - Створення задачі оптимізації (HTTP 202)
  - Результату оптимізації з деталізованим розкладом маршруту
  - Статуси виконання задач (PENDING → RUNNING → COMPLETED / FAILED)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Статус задачі оптимізації
# ---------------------------------------------------------------------------


class TaskStatusEnum(str, Enum):
    """Статус виконання задачі оптимізації.

    Перехід станів: PENDING → RUNNING → COMPLETED | FAILED
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Деталізований розклад для однієї точки маршруту
# ---------------------------------------------------------------------------


class ScheduledTaskItem(BaseModel):
    """Результат планування для однієї точки маршруту.

    Всі часові значення — у хвилинах від початку доби.

    Attributes
    ----------
    task_id : str
        Ідентифікатор завдання (відповідає Task.id з вхідного запиту).
    arrival_time : float
        Фактичний час прибуття aᵢ (хвилини від початку доби).
    wait_time : float
        Час очікування до відкриття часового вікна max(0, eᵢ − aᵢ).
    service_start : float
        Час початку обслуговування max(aᵢ, eᵢ).
    lateness : float
        Запізнення max(0, aᵢ − lᵢ) (хвилини).
    departure_time : float
        Час виїзду з точки після обслуговування.
    """

    task_id: str = Field(
        ...,
        description="Ідентифікатор завдання з вхідного запиту",
    )
    arrival_time: float = Field(
        ...,
        description="Час прибуття (хвилини від початку доби)",
    )
    wait_time: float = Field(
        ...,
        ge=0.0,
        description="Час очікування до відкриття часового вікна (хвилини)",
    )
    service_start: float = Field(
        ...,
        description="Час початку обслуговування (хвилини від початку доби)",
    )
    lateness: float = Field(
        ...,
        ge=0.0,
        description="Запізнення відносно дедлайну (хвилини)",
    )
    departure_time: float = Field(
        ...,
        description="Час виїзду з точки (хвилини від початку доби)",
    )


# ---------------------------------------------------------------------------
# Відповідь при створенні задачі (HTTP 202 Accepted)
# ---------------------------------------------------------------------------


class OptimizationTaskCreateResponse(BaseModel):
    """Відповідь при постановці задачі оптимізації у чергу.

    Повертається з HTTP 202 Accepted.
    """

    task_id: UUID = Field(
        ...,
        description="Унікальний ідентифікатор задачі (UUID4)",
    )
    status: TaskStatusEnum = Field(
        default=TaskStatusEnum.PENDING,
        description="Початковий статус задачі",
    )
    message: str = Field(
        default="Задачу прийнято до обробки",
        description="Повідомлення для клієнта",
    )


# ---------------------------------------------------------------------------
# Повний результат оптимізації
# ---------------------------------------------------------------------------


class OptimizationResultResponse(BaseModel):
    """Повний результат оптимізації маршруту.

    Включає статус виконання, метрики маршруту, деталізований розклад
    та історії збіжності для аналізу та побудови графіків.
    """

    task_id: UUID = Field(
        ...,
        description="Унікальний ідентифікатор задачі (UUID4)",
    )
    status: TaskStatusEnum = Field(
        ...,
        description="Поточний статус задачі",
    )
    total_duration: Optional[float] = Field(
        default=None,
        description="Загальний час маршруту T_total (хвилини)",
    )
    total_lateness: Optional[float] = Field(
        default=None,
        description="Сумарне запізнення (хвилини)",
    )
    fitness_cost: Optional[float] = Field(
        default=None,
        description="Фінальне значення фітнес-функції F(Route)",
    )
    scheduled_route: Optional[List[ScheduledTaskItem]] = Field(
        default=None,
        description="Деталізований розклад маршруту по кожній точці",
    )
    convergence_history: Optional[List[float]] = Field(
        default=None,
        description="Значення найкращого cost по кожному поколінню GA",
    )
    mutation_rate_history: Optional[List[float]] = Field(
        default=None,
        description="Історія зміни P_m по поколіннях",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Повідомлення про помилку (якщо status == FAILED)",
    )

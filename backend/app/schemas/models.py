"""
Доменні моделі для задачі TD-VRPTW-P (Time-Dependent Vehicle Routing Problem
with Time Windows and Priorities).

Визначає Pydantic v2 моделі для валідації вхідних даних оптимізатора маршрутів
відповідно до математичної постановки:
  - G = (V, E) — зважений орієнтований граф локацій
  - Часові вікна [eᵢ, lᵢ], тривалість обслуговування sᵢ, пріоритет pᵢ ∈ [1,5]
  - Вагові коефіцієнти фітнес-функції w₁, w₂, w₃
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Location  (вершина графа vᵢ)
# ---------------------------------------------------------------------------

class Location(BaseModel):
    """Географічна точка з WGS-84 координатами.

    Відповідає вершині vᵢ ∈ V у графі маршрутів.  Координати валідуються
    на відповідність діапазонам: широта [-90, 90], довгота [-180, 180].
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        ...,
        min_length=1,
        description="Унікальний ідентифікатор локації",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Географічна широта (WGS-84), діапазон [-90.0, 90.0]",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Географічна довгота (WGS-84), діапазон [-180.0, 180.0]",
    )
    name: Optional[str] = Field(
        default=None,
        description="Зрозуміла людині назва локації (опціонально)",
    )


# ---------------------------------------------------------------------------
# TimeWindow  (часове вікно [eᵢ, lᵢ])
# ---------------------------------------------------------------------------

class TimeWindow(BaseModel):
    """Часове вікно обслуговування [start_time, end_time].

    Підтримує як UNIX-мітки часу (int, секунди), так і ISO-8601 datetime.
    Інваріант: start_time ≤ end_time.
    """

    model_config = ConfigDict(frozen=True)

    start_time: Union[int, datetime] = Field(
        ...,
        description=(
            "Найраніший допустимий час початку обслуговування (eᵢ). "
            "Ціле число (UNIX timestamp, секунди) або datetime."
        ),
    )
    end_time: Union[int, datetime] = Field(
        ...,
        description=(
            "Крайній допустимий час завершення обслуговування (lᵢ). "
            "Ціле число (UNIX timestamp, секунди) або datetime."
        ),
    )

    @model_validator(mode="after")
    def _check_window_order(self) -> "TimeWindow":
        """Перевіряє, що start_time ≤ end_time.

        Порівняння виконується між значеннями одного типу (int з int,
        datetime з datetime).  Якщо типи відрізняються — це помилка
        конфігурації, і валідатор повідомить про несумісність.
        """
        s, e = self.start_time, self.end_time

        if type(s) is not type(e):
            raise ValueError(
                "start_time та end_time мають бути одного типу "
                f"(отримано {type(s).__name__} і {type(e).__name__})"
            )

        if s > e:
            raise ValueError(
                f"start_time ({s}) не може бути пізніше за end_time ({e}): "
                "часове вікно інвертоване"
            )

        return self


# ---------------------------------------------------------------------------
# Task  (завдання на обслуговування в точці vᵢ)
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """Завдання, яке необхідно виконати у певній локації.

    Кожному завданню відповідає вершина графа vᵢ ∈ V \\ {v₀} з параметрами:
      - Часове вікно [eᵢ, lᵢ]
      - Тривалість обслуговування sᵢ
      - Пріоритет pᵢ ∈ [1, 5]
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        ...,
        min_length=1,
        description="Унікальний ідентифікатор завдання",
    )
    location: Location = Field(
        ...,
        description="Локація виконання завдання (вершина графа vᵢ)",
    )
    time_window: TimeWindow = Field(
        ...,
        description="Часове вікно обслуговування [eᵢ, lᵢ]",
    )
    service_duration: int = Field(
        ...,
        gt=0,
        description=(
            "Тривалість обслуговування sᵢ (у секундах). "
            "Має бути строго додатнім."
        ),
    )
    priority: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Пріоритет завдання pᵢ ∈ [1, 5], де 1 — найнижчий, "
            "5 — найвищий пріоритет."
        ),
    )


# ---------------------------------------------------------------------------
# OptimizationRequest  (повний запит до оптимізатора)
# ---------------------------------------------------------------------------

class OptimizationRequest(BaseModel):
    """Вхідні дані для запуску оптимізації маршруту.

    Об'єднує:
      - Депо (початкова точка v₀)
      - Список завдань (вершини v₁…vₙ)
      - Час старту маршруту
      - Вагові коефіцієнти фітнес-функції F(Route) = w₁·T_total + w₂·Σpenalty + w₃·Σunvisited
    """

    model_config = ConfigDict(frozen=True)

    depot: Location = Field(
        ...,
        description="Початкова точка маршруту (депо, v₀)",
    )
    tasks: List[Task] = Field(
        ...,
        min_length=1,
        description=(
            "Список завдань для оптимізації. "
            "Має містити щонайменше одне завдання."
        ),
    )
    start_time: Union[int, datetime] = Field(
        ...,
        description=(
            "Час початку маршруту. "
            "Ціле число (UNIX timestamp, секунди) або datetime."
        ),
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Вагові коефіцієнти фітнес-функції: "
            "{'w1': ..., 'w2': ..., 'w3': ...}. "
            "w1 — вага загального часу, "
            "w2 — вага штрафу за запізнення, "
            "w3 — вага штрафу за невідвідані точки. "
            "Якщо None — використовуються значення за замовчуванням."
        ),
    )

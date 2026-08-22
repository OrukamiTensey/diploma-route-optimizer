"""
Менеджер фонових задач оптимізації маршрутів.

Забезпечує:
  - Потокобезпечне in-memory сховище стану задач (Dict[UUID, ...])
  - Запуск CPU-bound GeneticOptimizer.optimize() у пулі потоків asyncio
    (не блокує event loop FastAPI)
  - Побудову деталізованого розкладу ScheduledTaskItem з результатів GA
  - Перехід станів: PENDING → RUNNING → COMPLETED | FAILED
"""

from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from app.core.ga import GeneticOptimizer, OptimizationResult, RouteSimulation
from app.schemas.models import OptimizationRequest, Task
from app.schemas.responses import (
    OptimizationResultResponse,
    OptimizationTaskCreateResponse,
    ScheduledTaskItem,
    TaskStatusEnum,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Побудова деталізованого розкладу маршруту
# ---------------------------------------------------------------------------


def _build_scheduled_route(
    best_route: List[Task],
    simulation: RouteSimulation,
    start_time_min: float,
) -> List[ScheduledTaskItem]:
    """Будує список ScheduledTaskItem з результатів симуляції.

    Parameters
    ----------
    best_route : List[Task]
        Оптимальна послідовність завдань.
    simulation : RouteSimulation
        Результат симуляції маршруту (arrival/departure times).
    start_time_min : float
        Час старту маршруту (хвилини від початку доби).

    Returns
    -------
    List[ScheduledTaskItem]
        Деталізований розклад по кожній точці маршруту.
    """
    items: List[ScheduledTaskItem] = []

    for i, task in enumerate(best_route):
        arrival = simulation.arrival_times[i]
        departure = simulation.departure_times[i]

        # Часове вікно у хвилинах
        tw_start = _tw_to_minutes(task.time_window.start_time)
        tw_end = _tw_to_minutes(task.time_window.end_time)

        # Очікування до відкриття вікна
        wait_time = max(0.0, tw_start - arrival)

        # Початок обслуговування
        service_start = max(arrival, tw_start)

        # Запізнення
        lateness = max(0.0, arrival - tw_end)

        items.append(
            ScheduledTaskItem(
                task_id=task.id,
                arrival_time=round(arrival, 2),
                wait_time=round(wait_time, 2),
                service_start=round(service_start, 2),
                lateness=round(lateness, 2),
                departure_time=round(departure, 2),
            )
        )

    return items


def _tw_to_minutes(value: object) -> float:
    """Конвертує значення часового вікна в хвилини від початку доби."""
    from datetime import datetime

    if isinstance(value, int):
        return float(value)
    elif isinstance(value, datetime):
        return value.hour * 60.0 + value.minute + value.second / 60.0
    else:
        return float(value)  # type: ignore[arg-type]


def _resolve_start_time(start_time: object) -> float:
    """Конвертує start_time у хвилини від початку доби."""
    from datetime import datetime

    if isinstance(start_time, int):
        return float(start_time)
    elif isinstance(start_time, datetime):
        return start_time.hour * 60.0 + start_time.minute + start_time.second / 60.0
    else:
        return float(start_time)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------


class TaskManager:
    """Потокобезпечний менеджер фонових задач оптимізації.

    Зберігає стан та результати задач в in-memory словнику.
    CPU-bound оптимізація запускається через ``asyncio.to_thread()``
    (Python 3.9+) або ``loop.run_in_executor()`` для сумісності.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[UUID, OptimizationResultResponse] = {}

    async def submit(
        self,
        request: OptimizationRequest,
        *,
        generations: int = 100,
        pop_size: int = 60,
        seed: Optional[int] = None,
    ) -> OptimizationTaskCreateResponse:
        """Ставить задачу оптимізації у чергу на виконання.

        Parameters
        ----------
        request : OptimizationRequest
            Вхідні дані для оптимізації.
        generations : int
            Кількість поколінь GA.
        pop_size : int
            Розмір популяції.
        seed : Optional[int]
            Зерно RNG для відтворюваності.

        Returns
        -------
        OptimizationTaskCreateResponse
            Ідентифікатор задачі та початковий статус PENDING.
        """
        task_id = uuid4()

        # Реєструємо задачу зі статусом PENDING
        initial_state = OptimizationResultResponse(
            task_id=task_id,
            status=TaskStatusEnum.PENDING,
        )
        with self._lock:
            self._tasks[task_id] = initial_state

        # Запускаємо обчислення у фоні (не блокуючи event loop)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,  # default thread pool
            self._run_optimization,
            task_id,
            request,
            generations,
            pop_size,
            seed,
        )

        logger.info("Task %s submitted (PENDING)", task_id)

        return OptimizationTaskCreateResponse(
            task_id=task_id,
            status=TaskStatusEnum.PENDING,
            message="Задачу прийнято до обробки",
        )

    def get_result(self, task_id: UUID) -> Optional[OptimizationResultResponse]:
        """Повертає поточний стан/результат задачі.

        Parameters
        ----------
        task_id : UUID
            Ідентифікатор задачі.

        Returns
        -------
        Optional[OptimizationResultResponse]
            Стан задачі або None, якщо задачі не існує.
        """
        with self._lock:
            return self._tasks.get(task_id)

    def _run_optimization(
        self,
        task_id: UUID,
        request: OptimizationRequest,
        generations: int,
        pop_size: int,
        seed: Optional[int],
    ) -> None:
        """Виконує оптимізацію у робочому потоці (thread pool).

        Оновлює стан задачі: PENDING → RUNNING → COMPLETED | FAILED.
        """
        # Переводимо в RUNNING
        with self._lock:
            self._tasks[task_id] = OptimizationResultResponse(
                task_id=task_id,
                status=TaskStatusEnum.RUNNING,
            )

        logger.info("Task %s started (RUNNING)", task_id)

        try:
            optimizer = GeneticOptimizer(
                request,
                generations=generations,
                pop_size=pop_size,
                seed=seed,
            )
            result: OptimizationResult = optimizer.optimize()

            # Побудова деталізованого розкладу
            start_time_min = _resolve_start_time(request.start_time)

            # Отримуємо хромосому найкращого маршруту для симуляції
            best_chromosome = [
                next(
                    idx
                    for idx, t in enumerate(list(request.tasks))
                    if t.id == route_task.id
                )
                for route_task in result.best_route
            ]
            best_sim: RouteSimulation = optimizer.simulate(best_chromosome)

            scheduled_route = _build_scheduled_route(
                best_route=result.best_route,
                simulation=best_sim,
                start_time_min=start_time_min,
            )

            # Зберігаємо COMPLETED
            with self._lock:
                self._tasks[task_id] = OptimizationResultResponse(
                    task_id=task_id,
                    status=TaskStatusEnum.COMPLETED,
                    total_duration=round(result.total_time, 2),
                    total_lateness=round(result.total_lateness, 2),
                    fitness_cost=round(result.cost, 4),
                    scheduled_route=scheduled_route,
                    convergence_history=result.convergence_history,
                    mutation_rate_history=result.mutation_rate_history,
                )

            logger.info(
                "Task %s completed (cost=%.4f, time=%.2f min)",
                task_id,
                result.cost,
                result.total_time,
            )

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Task %s failed: %s\n%s", task_id, exc, tb)

            with self._lock:
                self._tasks[task_id] = OptimizationResultResponse(
                    task_id=task_id,
                    status=TaskStatusEnum.FAILED,
                    error_message=str(exc),
                )


# ---------------------------------------------------------------------------
# Глобальний екземпляр (Singleton)
# ---------------------------------------------------------------------------

task_manager = TaskManager()

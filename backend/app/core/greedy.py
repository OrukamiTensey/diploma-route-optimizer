"""
Жадібний алгоритм (Greedy / Nearest Neighbor) для задачі TD-VRPTW-P.

Евристика найменшого інкрементального cost: на кожному кроці обирається
невідвідана точка, додавання якої до маршруту дає мінімальний приріст
загальної фітнес-функції з урахуванням:
  - Часу переїзду t_ij(T) з динамічним трафіком
  - Очікування до відкриття часового вікна
  - Штрафу за запізнення поза часове вікно
  - Тривалості обслуговування

Використовується як baseline (бейзлайн) для бенчмаркінгу GA та Hybrid GA.

Відповідає специфікації: docs/specs/ALGORITHMS_SPEC.md
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.ga import DEFAULT_WEIGHTS, OptimizationResult, RouteSimulation
from app.core.traffic import TrafficMatrixGenerator
from app.schemas.models import Location, OptimizationRequest, Task


class GreedyOptimizer:
    """Жадібний оптимізатор маршрутів для TD-VRPTW-P.

    На кожному кроці вибирає наступну невідвідану точку за критерієм
    мінімального інкрементального cost, що враховує час доїзду з
    динамічним трафіком, очікування та штраф за дедлайн.

    Parameters
    ----------
    request : OptimizationRequest
        Запит на оптимізацію (депо, завдання, час старту, ваги).
    """

    def __init__(self, request: OptimizationRequest) -> None:
        self._depot: Location = request.depot
        self._tasks: List[Task] = list(request.tasks)
        self._n: int = len(self._tasks)
        self._start_time_min: float = self._resolve_start_time(
            request.start_time,
        )
        self._weights: Dict[str, float] = dict(
            request.weights if request.weights else DEFAULT_WEIGHTS
        )
        self._traffic: TrafficMatrixGenerator = TrafficMatrixGenerator()

    # -- публічний API ------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """Запускає жадібну евристику та повертає результат.

        Returns
        -------
        OptimizationResult
            Маршрут, побудований за Greedy-евристикою, з метриками
            у форматі, сумісному з GA.
        """
        # Побудова жадібного порядку
        greedy_chromosome = self._build_greedy_route()

        # Симуляція повного маршруту для отримання точних метрик
        sim = self._simulate(greedy_chromosome)
        cost = self._evaluate_full(greedy_chromosome, sim)

        best_route = [self._tasks[i] for i in greedy_chromosome]

        return OptimizationResult(
            best_route=best_route,
            total_time=sim.total_time,
            total_lateness=sim.total_lateness,
            cost=cost,
            convergence_history=[cost],
            mutation_rate_history=[],
        )

    # -- жадібна побудова маршруту ------------------------------------------

    def _build_greedy_route(self) -> List[int]:
        """Будує маршрут за евристикою найменшого інкрементального cost.

        На кожному кроці:
        1. Для кожної невідвіданої точки обчислюється інкрементальний cost.
        2. Обирається точка з мінімальним cost.
        3. Оновлюється поточна позиція та час.

        Returns
        -------
        List[int]
            Перестановка індексів завдань — порядок відвідування.
        """
        visited: List[int] = []
        unvisited = set(range(self._n))

        current_loc = self._depot
        current_time = self._start_time_min

        w1 = self._weights.get("w1", 1.0)
        w2 = self._weights.get("w2", 50.0)

        while unvisited:
            best_idx: Optional[int] = None
            best_cost = float("inf")
            best_depart_time = current_time

            for idx in unvisited:
                task = self._tasks[idx]

                # Час переїзду з урахуванням трафіку
                travel_time = self._traffic.get_travel_time(
                    current_loc, task.location, departure_time=current_time,
                )

                # Час прибуття
                arrival = current_time + travel_time

                # Часове вікно (хвилини)
                tw_start = self._tw_to_minutes(task.time_window.start_time)
                tw_end = self._tw_to_minutes(task.time_window.end_time)

                # Очікування
                wait = max(0.0, tw_start - arrival)

                # Запізнення
                lateness = max(0.0, arrival - tw_end)

                # Тривалість обслуговування (секунди → хвилини)
                service_min = task.service_duration / 60.0

                # Час виїзду з цієї точки
                depart = max(arrival, tw_start) + service_min

                # Інкрементальний cost
                incremental_cost = (
                    w1 * (travel_time + wait + service_min)
                    + w2 * lateness
                )

                if incremental_cost < best_cost:
                    best_cost = incremental_cost
                    best_idx = idx
                    best_depart_time = depart

            assert best_idx is not None  # unvisited не порожній
            visited.append(best_idx)
            unvisited.remove(best_idx)
            current_loc = self._tasks[best_idx].location
            current_time = best_depart_time

        return visited

    # -- симуляція та оцінка (делегування до GA-логіки) ---------------------

    def _simulate(self, chromosome: List[int]) -> RouteSimulation:
        """Симулює маршрут depot → tasks[ch[0]] → … → depot.

        Повна часова симуляція з урахуванням t_ij(T), очікувань,
        запізнень та обслуговування.
        """
        arrival_times: List[float] = []
        departure_times: List[float] = []
        total_lateness: float = 0.0

        current_loc = self._depot
        current_time = self._start_time_min

        for task_idx in chromosome:
            task = self._tasks[task_idx]

            travel_time = self._traffic.get_travel_time(
                current_loc, task.location, departure_time=current_time,
            )

            arrival = current_time + travel_time
            arrival_times.append(arrival)

            tw_start = self._tw_to_minutes(task.time_window.start_time)
            tw_end = self._tw_to_minutes(task.time_window.end_time)

            service_start = max(arrival, tw_start)
            lateness = max(0.0, arrival - tw_end)
            total_lateness += lateness

            service_min = task.service_duration / 60.0
            depart = service_start + service_min
            departure_times.append(depart)

            current_loc = task.location
            current_time = depart

        # Повернення до депо
        if chromosome:
            return_travel = self._traffic.get_travel_time(
                current_loc, self._depot, departure_time=current_time,
            )
            current_time += return_travel

        total_time = current_time - self._start_time_min

        return RouteSimulation(
            total_time=total_time,
            total_lateness=total_lateness,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )

    def _evaluate_full(
        self, chromosome: List[int], sim: RouteSimulation,
    ) -> float:
        """Обчислює повну фітнес-функцію F(Route).

        F = w₁·T_total + w₂·Σlateness + w₃·Σp_j(unvisited)
        """
        w1 = self._weights.get("w1", 1.0)
        w2 = self._weights.get("w2", 50.0)
        w3 = self._weights.get("w3", 100.0)

        visited_set = set(chromosome)
        unvisited_priority = sum(
            self._tasks[i].priority
            for i in range(self._n)
            if i not in visited_set
        )

        return (
            w1 * sim.total_time
            + w2 * sim.total_lateness
            + w3 * unvisited_priority
        )

    # -- утиліти ------------------------------------------------------------

    @staticmethod
    def _resolve_start_time(start_time: object) -> float:
        """Конвертує start_time у хвилини від початку доби."""
        from datetime import datetime

        if isinstance(start_time, int):
            return float(start_time)
        elif isinstance(start_time, datetime):
            return (
                start_time.hour * 60.0
                + start_time.minute
                + start_time.second / 60.0
            )
        else:
            return float(start_time)  # type: ignore[arg-type]

    @staticmethod
    def _tw_to_minutes(value: object) -> float:
        """Конвертує значення часового вікна в хвилини від початку доби."""
        from datetime import datetime

        if isinstance(value, int):
            return float(value)
        elif isinstance(value, datetime):
            return value.hour * 60.0 + value.minute + value.second / 60.0
        else:
            return float(value)  # type: ignore[arg-type]

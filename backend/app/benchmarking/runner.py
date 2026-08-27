"""
Модуль проведення бенчмаркінгових експериментів для TD-VRPTW-P.

Забезпечує:
  - Генерацію тестових сценаріїв різної складності (10/25/50 задач)
    з різною щільністю часових вікон та рівнями заторів.
  - Запуск трьох алгоритмів: Greedy, Standard GA, Hybrid GA + 2-opt.
  - Збір метрик: execution_time, fitness_cost, travel_time, lateness,
    convergence_curve.
  - Експорт у Pandas DataFrame / CSV / JSON.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.ga import GeneticOptimizer, OptimizationResult
from app.core.greedy import GreedyOptimizer
from app.schemas.models import Location, OptimizationRequest, Task, TimeWindow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Метрики одного запуску алгоритму
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmMetrics:
    """Метрики одного запуску алгоритму на одному сценарії.

    Attributes
    ----------
    algorithm : str
        Назва алгоритму (``greedy``, ``standard_ga``, ``hybrid_ga``).
    scenario : str
        Назва сценарію (наприклад, ``small_normal``).
    num_tasks : int
        Кількість завдань у сценарії.
    execution_time_ms : float
        Час виконання (мілісекунди, CPU clock).
    fitness_cost : float
        Значення фітнес-функції F(Route).
    total_travel_time : float
        Загальний час маршруту T_total (хвилини).
    total_lateness : float
        Сумарне запізнення (хвилини).
    convergence_curve : List[float]
        Історія збіжності фітнесу (для GA — по поколіннях).
    """

    algorithm: str
    scenario: str
    num_tasks: int
    execution_time_ms: float
    fitness_cost: float
    total_travel_time: float
    total_lateness: float
    convergence_curve: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Опис сценарію
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Конфігурація одного тестового сценарію.

    Attributes
    ----------
    name : str
        Унікальна назва сценарію.
    num_tasks : int
        Кількість завдань.
    tight_windows : bool
        True — вузькі часові вікна, False — широкі.
    start_time_min : float
        Час старту маршруту (хвилини від початку доби).
    """

    name: str
    num_tasks: int
    tight_windows: bool
    start_time_min: float


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


# Стандартні сценарії
DEFAULT_SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig("small_normal", 10, False, 540.0),
    ScenarioConfig("small_tight", 10, True, 540.0),
    ScenarioConfig("medium_normal", 25, False, 720.0),
    ScenarioConfig("medium_tight", 25, True, 540.0),
    ScenarioConfig("large_normal", 50, False, 720.0),
    ScenarioConfig("large_tight", 50, True, 1080.0),
]


class BenchmarkRunner:
    """Модуль проведення порівняльних експериментів.

    Генерує тестові сценарії, запускає три алгоритми (Greedy,
    Standard GA, Hybrid GA+2-opt) та збирає метрики для аналізу.

    Parameters
    ----------
    scenarios : Optional[List[ScenarioConfig]]
        Список сценаріїв для тестування.  Якщо ``None`` —
        використовуються ``DEFAULT_SCENARIOS``.
    ga_generations : int
        Кількість поколінь GA.
    ga_pop_size : int
        Розмір популяції GA.
    seed : Optional[int]
        Зерно RNG для відтворюваності.
    """

    # Координати центру Києва для генерації локацій
    _KYIV_CENTER_LAT: float = 50.45
    _KYIV_CENTER_LON: float = 30.52
    _COORD_SPREAD: float = 0.05

    def __init__(
        self,
        scenarios: Optional[List[ScenarioConfig]] = None,
        *,
        ga_generations: int = 100,
        ga_pop_size: int = 60,
        seed: Optional[int] = None,
    ) -> None:
        self._scenarios = scenarios or DEFAULT_SCENARIOS
        self._ga_generations = ga_generations
        self._ga_pop_size = ga_pop_size
        self._seed = seed
        self._rng = random.Random(seed)
        self._results: List[AlgorithmMetrics] = []

    # -- публічний API ------------------------------------------------------

    def generate_request(self, config: ScenarioConfig) -> OptimizationRequest:
        """Генерує ``OptimizationRequest`` для заданого сценарію.

        Parameters
        ----------
        config : ScenarioConfig
            Конфігурація сценарію.

        Returns
        -------
        OptimizationRequest
            Запит із згенерованим депо та завданнями.
        """
        depot = Location(
            id="depot",
            latitude=self._KYIV_CENTER_LAT,
            longitude=self._KYIV_CENTER_LON,
            name="Депо (центр Києва)",
        )

        tasks: List[Task] = []
        for i in range(config.num_tasks):
            lat = self._KYIV_CENTER_LAT + self._rng.uniform(
                -self._COORD_SPREAD, self._COORD_SPREAD,
            )
            lon = self._KYIV_CENTER_LON + self._rng.uniform(
                -self._COORD_SPREAD, self._COORD_SPREAD,
            )

            location = Location(
                id=f"loc-{i}",
                latitude=round(lat, 6),
                longitude=round(lon, 6),
                name=f"Точка {i}",
            )

            # Часові вікна (хвилини від початку доби)
            base_start = config.start_time_min + self._rng.uniform(0, 120)
            if config.tight_windows:
                window_width = self._rng.uniform(30, 60)  # Вузькі: 30-60 хв
            else:
                window_width = self._rng.uniform(120, 300)  # Широкі: 2-5 год

            tw_start = int(base_start)
            tw_end = int(base_start + window_width)

            # Тривалість обслуговування: 5–30 хвилин (у секундах)
            service_duration = self._rng.randint(5, 30) * 60

            # Пріоритет: 1–5
            priority = self._rng.randint(1, 5)

            tasks.append(
                Task(
                    id=f"task-{i}",
                    location=location,
                    time_window=TimeWindow(
                        start_time=tw_start, end_time=tw_end,
                    ),
                    service_duration=service_duration,
                    priority=priority,
                )
            )

        return OptimizationRequest(
            depot=depot,
            tasks=tasks,
            start_time=int(config.start_time_min),
        )

    def run_all(self) -> List[AlgorithmMetrics]:
        """Запускає всі алгоритми на всіх сценаріях.

        Returns
        -------
        List[AlgorithmMetrics]
            Зведений список метрик усіх запусків.
        """
        self._results = []

        for config in self._scenarios:
            logger.info(
                "Сценарій '%s': %d задач, %s вікна",
                config.name,
                config.num_tasks,
                "вузькі" if config.tight_windows else "широкі",
            )

            request = self.generate_request(config)

            # 1. Greedy
            metrics_greedy = self._run_greedy(request, config.name)
            self._results.append(metrics_greedy)
            logger.info(
                "  Greedy: cost=%.2f, time=%.1f ms",
                metrics_greedy.fitness_cost,
                metrics_greedy.execution_time_ms,
            )

            # 2. Standard GA (без локального пошуку)
            metrics_std_ga = self._run_ga(
                request, config.name,
                algorithm_name="standard_ga",
                enable_local_search=False,
            )
            self._results.append(metrics_std_ga)
            logger.info(
                "  Standard GA: cost=%.2f, time=%.1f ms",
                metrics_std_ga.fitness_cost,
                metrics_std_ga.execution_time_ms,
            )

            # 3. Hybrid GA + 2-opt
            metrics_hybrid = self._run_ga(
                request, config.name,
                algorithm_name="hybrid_ga",
                enable_local_search=True,
            )
            self._results.append(metrics_hybrid)
            logger.info(
                "  Hybrid GA: cost=%.2f, time=%.1f ms",
                metrics_hybrid.fitness_cost,
                metrics_hybrid.execution_time_ms,
            )

        return self._results

    def run_single(
        self, request: OptimizationRequest, scenario_name: str = "custom",
    ) -> List[AlgorithmMetrics]:
        """Запускає всі 3 алгоритми на одному запиті.

        Parameters
        ----------
        request : OptimizationRequest
            Запит на оптимізацію.
        scenario_name : str
            Назва сценарію для звітності.

        Returns
        -------
        List[AlgorithmMetrics]
            Метрики трьох алгоритмів.
        """
        results: List[AlgorithmMetrics] = []

        results.append(self._run_greedy(request, scenario_name))
        results.append(
            self._run_ga(
                request, scenario_name,
                algorithm_name="standard_ga",
                enable_local_search=False,
            )
        )
        results.append(
            self._run_ga(
                request, scenario_name,
                algorithm_name="hybrid_ga",
                enable_local_search=True,
            )
        )

        self._results.extend(results)
        return results

    # -- експорт результатів ------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Повертає зведену таблицю метрик як Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Таблиця з колонками: algorithm, scenario, num_tasks,
            execution_time_ms, fitness_cost, total_travel_time,
            total_lateness.
        """
        records = []
        for m in self._results:
            records.append({
                "algorithm": m.algorithm,
                "scenario": m.scenario,
                "num_tasks": m.num_tasks,
                "execution_time_ms": round(m.execution_time_ms, 2),
                "fitness_cost": round(m.fitness_cost, 4),
                "total_travel_time": round(m.total_travel_time, 2),
                "total_lateness": round(m.total_lateness, 2),
            })
        return pd.DataFrame(records)

    def to_csv(self, path: str | Path) -> None:
        """Зберігає результати у CSV-файл.

        Parameters
        ----------
        path : str | Path
            Шлях до файлу.
        """
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Результати збережено у %s", path)

    def to_json(self, path: str | Path) -> None:
        """Зберігає результати у JSON-файл.

        Parameters
        ----------
        path : str | Path
            Шлях до файлу.
        """
        data: List[Dict[str, Any]] = []
        for m in self._results:
            entry = asdict(m)
            entry["execution_time_ms"] = round(m.execution_time_ms, 2)
            entry["fitness_cost"] = round(m.fitness_cost, 4)
            data.append(entry)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Результати збережено у %s", path)

    @property
    def results(self) -> List[AlgorithmMetrics]:
        """Список зібраних метрик."""
        return list(self._results)

    # -- внутрішні методи ---------------------------------------------------

    def _run_greedy(
        self, request: OptimizationRequest, scenario_name: str,
    ) -> AlgorithmMetrics:
        """Запускає Greedy та збирає метрики."""
        start = time.perf_counter()
        optimizer = GreedyOptimizer(request)
        result = optimizer.optimize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return AlgorithmMetrics(
            algorithm="greedy",
            scenario=scenario_name,
            num_tasks=len(request.tasks),
            execution_time_ms=elapsed_ms,
            fitness_cost=result.cost,
            total_travel_time=result.total_time,
            total_lateness=result.total_lateness,
            convergence_curve=result.convergence_history,
        )

    def _run_ga(
        self,
        request: OptimizationRequest,
        scenario_name: str,
        *,
        algorithm_name: str,
        enable_local_search: bool,
    ) -> AlgorithmMetrics:
        """Запускає GA (standard або hybrid) та збирає метрики."""
        start = time.perf_counter()
        optimizer = GeneticOptimizer(
            request,
            generations=self._ga_generations,
            pop_size=self._ga_pop_size,
            enable_local_search=enable_local_search,
            seed=self._seed,
        )
        result = optimizer.optimize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return AlgorithmMetrics(
            algorithm=algorithm_name,
            scenario=scenario_name,
            num_tasks=len(request.tasks),
            execution_time_ms=elapsed_ms,
            fitness_cost=result.cost,
            total_travel_time=result.total_time,
            total_lateness=result.total_lateness,
            convergence_curve=result.convergence_history,
        )

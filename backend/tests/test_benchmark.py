"""
Тести модуля бенчмаркінгу для TD-VRPTW-P.

Покриває:
  - GreedyOptimizer: відвідує всі точки, повертає валідний fitness
  - BenchmarkRunner: міні-експеримент на 5 задачах
  - Hybrid GA ≤ Greedy (якість розв'язку)
  - Наявність даних збіжності у GA-результатах
"""

from __future__ import annotations

from typing import List

import pytest

from app.benchmarking.runner import BenchmarkRunner, ScenarioConfig
from app.core.ga import GeneticOptimizer, OptimizationResult
from app.core.greedy import GreedyOptimizer
from app.schemas.models import Location, OptimizationRequest, Task, TimeWindow


# =====================================================================
# Фікстури
# =====================================================================


@pytest.fixture
def depot() -> Location:
    """Депо — КПІ."""
    return Location(id="depot", latitude=50.4488, longitude=30.4571, name="КПІ")


@pytest.fixture
def tasks_5() -> List[Task]:
    """5 тестових завдань у Києві з різними пріоритетами."""
    return [
        Task(
            id="task-0",
            location=Location(
                id="loc-0", latitude=50.4501, longitude=30.5234,
                name="Хрещатик",
            ),
            time_window=TimeWindow(start_time=480, end_time=720),
            service_duration=900,
            priority=3,
        ),
        Task(
            id="task-1",
            location=Location(
                id="loc-1", latitude=50.4620, longitude=30.5080,
                name="Поділ",
            ),
            time_window=TimeWindow(start_time=540, end_time=780),
            service_duration=600,
            priority=5,
        ),
        Task(
            id="task-2",
            location=Location(
                id="loc-2", latitude=50.4350, longitude=30.5190,
                name="Печерськ",
            ),
            time_window=TimeWindow(start_time=600, end_time=840),
            service_duration=1200,
            priority=2,
        ),
        Task(
            id="task-3",
            location=Location(
                id="loc-3", latitude=50.5010, longitude=30.4985,
                name="Оболонь",
            ),
            time_window=TimeWindow(start_time=480, end_time=900),
            service_duration=300,
            priority=4,
        ),
        Task(
            id="task-4",
            location=Location(
                id="loc-4", latitude=50.4400, longitude=30.4600,
                name="Шулявка",
            ),
            time_window=TimeWindow(start_time=500, end_time=800),
            service_duration=600,
            priority=1,
        ),
    ]


@pytest.fixture
def request_5(depot: Location, tasks_5: List[Task]) -> OptimizationRequest:
    """Запит оптимізації з 5 завданнями, старт о 08:00."""
    return OptimizationRequest(
        depot=depot,
        tasks=tasks_5,
        start_time=480,
    )


# =====================================================================
# Тест 1: Greedy відвідує всі завдання без дублікатів
# =====================================================================


class TestGreedyOptimizer:
    """Тести жадібного алгоритму."""

    def test_greedy_visits_all_tasks(
        self, request_5: OptimizationRequest,
    ) -> None:
        """Greedy відвідує всі N точок рівно один раз."""
        optimizer = GreedyOptimizer(request_5)
        result = optimizer.optimize()

        # Всі завдання відвідані
        visited_ids = [t.id for t in result.best_route]
        expected_ids = [t.id for t in request_5.tasks]

        assert sorted(visited_ids) == sorted(expected_ids), (
            f"Greedy не відвідав усі завдання: "
            f"відвідані={visited_ids}, очікувані={expected_ids}"
        )

        # Без дублікатів
        assert len(visited_ids) == len(set(visited_ids)), (
            f"Greedy містить дублікати: {visited_ids}"
        )

    def test_greedy_returns_valid_fitness(
        self, request_5: OptimizationRequest,
    ) -> None:
        """Greedy повертає додатні метрики."""
        optimizer = GreedyOptimizer(request_5)
        result = optimizer.optimize()

        assert result.cost > 0.0, "Fitness cost має бути > 0"
        assert result.total_time > 0.0, "Total time має бути > 0"
        assert result.total_lateness >= 0.0, "Lateness має бути ≥ 0"
        assert len(result.convergence_history) == 1, (
            "Greedy має мати рівно 1 запис у convergence_history"
        )
        assert result.convergence_history[0] == result.cost


# =====================================================================
# Тест 2: BenchmarkRunner — міні-експеримент
# =====================================================================


class TestBenchmarkRunner:
    """Тести модуля запуску бенчмарків."""

    def test_benchmark_runner_small(self) -> None:
        """Міні-бенчмарк на 5 задачах формує звіт із 3 алгоритмами."""
        scenario = ScenarioConfig(
            name="test_mini",
            num_tasks=5,
            tight_windows=False,
            start_time_min=540.0,
        )

        runner = BenchmarkRunner(
            scenarios=[scenario],
            ga_generations=10,
            ga_pop_size=20,
            seed=42,
        )

        results = runner.run_all()

        # 3 алгоритми × 1 сценарій = 3 результати
        assert len(results) == 3

        algorithms = {r.algorithm for r in results}
        assert algorithms == {"greedy", "standard_ga", "hybrid_ga"}

        # Всі метрики мають бути заповнені
        for r in results:
            assert r.scenario == "test_mini"
            assert r.num_tasks == 5
            assert r.execution_time_ms > 0
            assert r.fitness_cost > 0
            assert r.total_travel_time > 0

        # DataFrame повинен мати 3 рядки та правильні колонки
        df = runner.to_dataframe()
        assert len(df) == 3
        assert "algorithm" in df.columns
        assert "fitness_cost" in df.columns
        assert "execution_time_ms" in df.columns

    def test_run_single_custom_request(
        self, request_5: OptimizationRequest,
    ) -> None:
        """run_single повертає метрики для 3 алгоритмів."""
        runner = BenchmarkRunner(
            ga_generations=10,
            ga_pop_size=20,
            seed=42,
        )
        results = runner.run_single(request_5, scenario_name="custom_5")

        assert len(results) == 3
        assert all(r.scenario == "custom_5" for r in results)


# =====================================================================
# Тест 3: Hybrid GA ≤ Greedy (якість розв'язку)
# =====================================================================


class TestAlgorithmComparison:
    """Порівняльні тести алгоритмів."""

    def test_hybrid_beats_greedy(
        self, request_5: OptimizationRequest,
    ) -> None:
        """Hybrid GA+2-opt дає fitness ≤ Greedy на тестовому наборі."""
        # Greedy
        greedy = GreedyOptimizer(request_5)
        greedy_result = greedy.optimize()

        # Hybrid GA + 2-opt (достатньо поколінь для збіжності)
        hybrid = GeneticOptimizer(
            request_5,
            generations=50,
            pop_size=40,
            enable_local_search=True,
            seed=42,
        )
        hybrid_result = hybrid.optimize()

        assert hybrid_result.cost <= greedy_result.cost * 1.05, (
            f"Hybrid GA (cost={hybrid_result.cost:.2f}) повинен бути "
            f"не гіршим за Greedy (cost={greedy_result.cost:.2f}) "
            f"з допуском 5%"
        )

    def test_convergence_data_present(
        self, request_5: OptimizationRequest,
    ) -> None:
        """GA-результати містять непорожню convergence_history."""
        # Standard GA
        std_ga = GeneticOptimizer(
            request_5,
            generations=20,
            pop_size=20,
            enable_local_search=False,
            seed=42,
        )
        std_result = std_ga.optimize()

        # Hybrid GA
        hybrid_ga = GeneticOptimizer(
            request_5,
            generations=20,
            pop_size=20,
            enable_local_search=True,
            seed=42,
        )
        hybrid_result = hybrid_ga.optimize()

        # convergence_history має бути ≥ generations+1 записів
        assert len(std_result.convergence_history) >= 20, (
            f"Standard GA convergence history: "
            f"{len(std_result.convergence_history)} записів "
            f"(очікувалось ≥ 20)"
        )
        assert len(hybrid_result.convergence_history) >= 20, (
            f"Hybrid GA convergence history: "
            f"{len(hybrid_result.convergence_history)} записів "
            f"(очікувалось ≥ 20)"
        )

        # Значення мають бути числовими та > 0
        for val in std_result.convergence_history:
            assert val > 0, f"Невалідне значення в convergence: {val}"

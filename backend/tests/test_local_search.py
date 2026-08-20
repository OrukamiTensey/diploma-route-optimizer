"""
Тести модуля локального пошуку 2-opt.

Покриває:
  - Покращення маршруту з перехресними ребрами (зменшення cost)
  - Збереження довжини хромосоми та відсутність втрати/дублювання задач
  - Коректність на тривіальних входах (1-2 елементи)
  - Функція не погіршує вже оптимальний маршрут
  - Інтеграція з повною фітнес-функцією GA (time-dependent)
"""

from __future__ import annotations

from typing import List

import pytest

from app.core.ga import GeneticOptimizer
from app.core.local_search import apply_2opt
from app.schemas.models import Location, OptimizationRequest, Task, TimeWindow


# =====================================================================
# Фікстури
# =====================================================================


def _simple_distance_cost(chromosome: List[int]) -> float:
    """Спрощена фітнес-функція: сума абсолютних різниць послідовних елементів.

    Для перестановки [0, 1, 2, ..., N-1] cost мінімальний.
    Для перестановки з «перехрестями» cost більший.
    """
    if len(chromosome) < 2:
        return 0.0
    return float(sum(abs(chromosome[i] - chromosome[i + 1]) for i in range(len(chromosome) - 1)))


@pytest.fixture
def depot() -> Location:
    return Location(id="depot", latitude=50.4488, longitude=30.4571, name="КПІ")


@pytest.fixture
def kyiv_tasks_6() -> List[Task]:
    """6 завдань у Києві, розміщених приблизно в ряд з Півдня на Північ."""
    lats = [50.420, 50.435, 50.450, 50.465, 50.480, 50.500]
    return [
        Task(
            id=f"task-{i}",
            location=Location(id=f"loc-{i}", latitude=lats[i], longitude=30.50),
            time_window=TimeWindow(start_time=480, end_time=1200),
            service_duration=600,
            priority=3,
        )
        for i in range(6)
    ]


# =====================================================================
# Базові тести apply_2opt
# =====================================================================


class TestApply2optBasic:
    """Базові юніт-тести 2-opt."""

    def test_preserves_chromosome_length(self) -> None:
        """Довжина хромосоми не змінюється після 2-opt."""
        ch = [4, 2, 0, 3, 1, 5]
        improved, _ = apply_2opt(ch, _simple_distance_cost, max_iterations=10)
        assert len(improved) == len(ch)

    def test_preserves_all_genes(self) -> None:
        """Всі задачі присутні рівно один раз (немає дублікатів і пропусків)."""
        ch = [4, 2, 0, 3, 1, 5]
        improved, _ = apply_2opt(ch, _simple_distance_cost, max_iterations=10)
        assert sorted(improved) == sorted(ch)

    def test_no_duplicate_genes(self) -> None:
        """Перевіряємо відсутність дублікатів після 2-opt."""
        ch = [5, 0, 3, 1, 4, 2]
        improved, _ = apply_2opt(ch, _simple_distance_cost, max_iterations=20)
        assert len(set(improved)) == len(improved)

    def test_does_not_mutate_original(self) -> None:
        """2-opt повертає нову хромосому, оригінал не змінюється."""
        ch = [3, 0, 2, 1]
        original = list(ch)
        _ = apply_2opt(ch, _simple_distance_cost)
        assert ch == original

    def test_single_element(self) -> None:
        """Хромосома з одним елементом повертається як є."""
        improved, cost = apply_2opt([0], _simple_distance_cost)
        assert improved == [0]
        assert cost == 0.0

    def test_two_elements(self) -> None:
        """Хромосома з двома елементами."""
        improved, cost = apply_2opt([1, 0], _simple_distance_cost)
        assert sorted(improved) == [0, 1]
        assert cost == 1.0  # |a - b| = 1 для будь-якої перестановки [0,1]


# =====================================================================
# Тести покращення маршруту
# =====================================================================


class TestApply2optImprovement:
    """Тести на реальне покращення маршруту."""

    def test_removes_crossing_and_reduces_cost(self) -> None:
        """Маршрут із перехрестями [0, 3, 1, 4, 2, 5] має бути покращений.

        Оптимальний для _simple_distance_cost: [0, 1, 2, 3, 4, 5] (cost = 5)
        Початковий cost = |0-3| + |3-1| + |1-4| + |4-2| + |2-5| = 3+2+3+2+3 = 13
        """
        ch = [0, 3, 1, 4, 2, 5]
        initial_cost = _simple_distance_cost(ch)
        assert initial_cost == 13.0

        improved, improved_cost = apply_2opt(ch, _simple_distance_cost)
        assert improved_cost < initial_cost, (
            f"2-opt не покращив маршрут: {improved_cost} >= {initial_cost}"
        )

    def test_finds_optimal_for_sequential(self) -> None:
        """Для послідовної фітнес-функції 2-opt знаходить [0,1,2,..,N-1] або зворотне."""
        ch = [0, 5, 1, 4, 2, 3]
        improved, improved_cost = apply_2opt(ch, _simple_distance_cost, max_iterations=100)
        # Оптимум: cost = N-1 (послідовний або зворотний порядок)
        assert improved_cost == 5.0

    def test_already_optimal_not_worsened(self) -> None:
        """Вже оптимальний маршрут не погіршується."""
        ch = [0, 1, 2, 3, 4]
        initial_cost = _simple_distance_cost(ch)
        improved, improved_cost = apply_2opt(ch, _simple_distance_cost)
        assert improved_cost <= initial_cost

    def test_reverse_optimal_not_worsened(self) -> None:
        """Зворотний оптимальний маршрут не погіршується."""
        ch = [4, 3, 2, 1, 0]
        initial_cost = _simple_distance_cost(ch)
        improved, improved_cost = apply_2opt(ch, _simple_distance_cost)
        assert improved_cost <= initial_cost

    def test_max_iterations_respected(self) -> None:
        """При max_iterations=0 хромосома не змінюється."""
        ch = [0, 5, 1, 4, 2, 3]
        improved, _ = apply_2opt(ch, _simple_distance_cost, max_iterations=0)
        assert improved == ch


# =====================================================================
# Інтеграція з повною TD-VRPTW-P фітнес-функцією
# =====================================================================


class TestApply2optWithGA:
    """Тести 2-opt із реальною фітнес-функцією GA."""

    def test_reduces_cost_with_real_fitness(
        self, depot: Location, kyiv_tasks_6: List[Task]
    ) -> None:
        """2-opt зменшує cost при використанні повної фітнес-функції GA."""
        req = OptimizationRequest(depot=depot, tasks=kyiv_tasks_6, start_time=480)
        opt = GeneticOptimizer(
            req, generations=1, pop_size=2, seed=1, enable_local_search=False
        )

        # Субоптимальний порядок: завдання у зворотному географічному порядку
        bad_route = [5, 3, 1, 4, 0, 2]
        initial_cost = opt.evaluate(bad_route)

        improved, improved_cost = apply_2opt(bad_route, opt.evaluate, max_iterations=20)

        assert improved_cost <= initial_cost
        assert sorted(improved) == list(range(6))

    def test_preserves_valid_permutation_with_real_fitness(
        self, depot: Location, kyiv_tasks_6: List[Task]
    ) -> None:
        """Після 2-opt хромосома залишається валідною перестановкою."""
        req = OptimizationRequest(depot=depot, tasks=kyiv_tasks_6, start_time=480)
        opt = GeneticOptimizer(
            req, generations=1, pop_size=2, seed=1, enable_local_search=False
        )

        ch = [2, 5, 0, 4, 1, 3]
        improved, _ = apply_2opt(ch, opt.evaluate, max_iterations=10)

        assert len(improved) == 6
        assert set(improved) == set(range(6))

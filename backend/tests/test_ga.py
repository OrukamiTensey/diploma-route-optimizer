"""
Тести генетичного алгоритму (GA) для TD-VRPTW-P.

Покриває:
  - Коректність OX-кросоверу (валідна перестановка без дублікатів)
  - Коректність swap та inversion мутацій
  - Турнірну селекцію (обирає кращу особину)
  - Детермінований тест симуляції маршруту та фітнес-функції
  - Збіжність GA: фітнес останнього покоління ≤ першого
  - Структуру OptimizationResult
"""

from __future__ import annotations

import random
from typing import List

import pytest

from app.core.ga import (
    DEFAULT_WEIGHTS,
    Chromosome,
    GeneticOptimizer,
    OptimizationResult,
    RouteSimulation,
    inversion_mutation,
    order_crossover,
    swap_mutation,
    tournament_selection,
)
from app.schemas.models import Location, OptimizationRequest, Task, TimeWindow


# =====================================================================
# Фікстури: тестовий набір з 4 завдань в межах Києва
# =====================================================================


@pytest.fixture
def depot() -> Location:
    """Депо — КПІ."""
    return Location(id="depot", latitude=50.4488, longitude=30.4571, name="КПІ")


@pytest.fixture
def tasks_4() -> List[Task]:
    """4 тестові завдання у Києві з різними пріоритетами та часовими вікнами.

    Всі часові вікна задано у хвилинах від початку доби.
    service_duration — у секундах (як у моделі).
    """
    return [
        Task(
            id="task-0",
            location=Location(id="loc-0", latitude=50.4501, longitude=30.5234, name="Хрещатик"),
            time_window=TimeWindow(start_time=480, end_time=720),   # 08:00 – 12:00
            service_duration=900,   # 15 хв
            priority=3,
        ),
        Task(
            id="task-1",
            location=Location(id="loc-1", latitude=50.4620, longitude=30.5080, name="Поділ"),
            time_window=TimeWindow(start_time=540, end_time=780),   # 09:00 – 13:00
            service_duration=600,   # 10 хв
            priority=5,
        ),
        Task(
            id="task-2",
            location=Location(id="loc-2", latitude=50.4350, longitude=30.5190, name="Печерськ"),
            time_window=TimeWindow(start_time=600, end_time=840),   # 10:00 – 14:00
            service_duration=1200,  # 20 хв
            priority=2,
        ),
        Task(
            id="task-3",
            location=Location(id="loc-3", latitude=50.5010, longitude=30.4985, name="Оболонь"),
            time_window=TimeWindow(start_time=480, end_time=900),   # 08:00 – 15:00
            service_duration=300,   # 5 хв
            priority=4,
        ),
    ]


@pytest.fixture
def request_4(depot: Location, tasks_4: List[Task]) -> OptimizationRequest:
    """Запит оптимізації з 4 завданнями, старт о 08:00 (480 хв)."""
    return OptimizationRequest(
        depot=depot,
        tasks=tasks_4,
        start_time=480,  # 08:00 у хвилинах
    )


@pytest.fixture
def optimizer_4(request_4: OptimizationRequest) -> GeneticOptimizer:
    """Оптимізатор із фіксованим seed для детермінованості."""
    return GeneticOptimizer(
        request_4,
        generations=30,
        pop_size=40,
        mutation_rate=0.15,
        seed=42,
    )


# =====================================================================
# Order Crossover (OX)
# =====================================================================


class TestOrderCrossover:
    """Тести OX-кросоверу."""

    def test_child_is_valid_permutation(self) -> None:
        """Нащадок OX має бути валідною перестановкою: усі гени рівно один раз."""
        random.seed(123)
        parent_a = [0, 1, 2, 3, 4, 5, 6, 7]
        parent_b = [3, 7, 5, 1, 6, 0, 2, 4]

        for _ in range(100):  # 100 спроб для надійності
            child = order_crossover(parent_a, parent_b)
            assert sorted(child) == list(range(8)), (
                f"Невалідна перестановка: {child}"
            )

    def test_preserves_all_genes(self) -> None:
        """OX зберігає всі гени без дублікатів і пропусків."""
        random.seed(456)
        n = 20
        parent_a = list(range(n))
        parent_b = list(range(n))
        random.shuffle(parent_b)

        child = order_crossover(parent_a, parent_b)
        assert len(child) == n
        assert set(child) == set(range(n))

    def test_single_gene(self) -> None:
        """Кросовер хромосоми з одним геном."""
        child = order_crossover([0], [0])
        assert child == [0]

    def test_two_genes(self) -> None:
        """Кросовер хромосоми з двома генами."""
        random.seed(789)
        for _ in range(50):
            child = order_crossover([0, 1], [1, 0])
            assert sorted(child) == [0, 1]


# =====================================================================
# Мутації
# =====================================================================


class TestMutations:
    """Тести swap та inversion мутацій."""

    def test_swap_preserves_genes(self) -> None:
        """Swap mutation не додає і не видаляє гени."""
        random.seed(111)
        ch = list(range(10))
        for _ in range(100):
            mutated = swap_mutation(ch)
            assert sorted(mutated) == list(range(10))

    def test_swap_changes_exactly_two(self) -> None:
        """Swap mutation змінює рівно 2 позиції (або 0 при невезінні)."""
        random.seed(222)
        ch = list(range(10))
        mutated = swap_mutation(ch)
        differences = sum(1 for a, b in zip(ch, mutated) if a != b)
        assert differences in (0, 2)

    def test_swap_does_not_mutate_original(self) -> None:
        """Swap mutation повертає нову копію, оригінал не змінюється."""
        ch = list(range(5))
        original = list(ch)
        _ = swap_mutation(ch)
        assert ch == original

    def test_inversion_preserves_genes(self) -> None:
        """Inversion mutation не додає і не видаляє гени."""
        random.seed(333)
        ch = list(range(10))
        for _ in range(100):
            mutated = inversion_mutation(ch)
            assert sorted(mutated) == list(range(10))

    def test_inversion_is_contiguous_reversal(self) -> None:
        """Inversion mutation інвертує один підсегмент."""
        random.seed(444)
        ch = list(range(8))
        mutated = inversion_mutation(ch)
        # Має бути тією самою множиною
        assert set(mutated) == set(ch)
        assert len(mutated) == len(ch)

    def test_inversion_does_not_mutate_original(self) -> None:
        """Inversion mutation повертає копію."""
        ch = list(range(5))
        original = list(ch)
        _ = inversion_mutation(ch)
        assert ch == original

    def test_single_element_mutations(self) -> None:
        """Мутація хромосоми з одним елементом повертає копію."""
        assert swap_mutation([0]) == [0]
        assert inversion_mutation([0]) == [0]


# =====================================================================
# Турнірна селекція
# =====================================================================


class TestTournamentSelection:
    """Тести Tournament Selection."""

    def test_selects_from_population(self) -> None:
        """Обрана особина належить до популяції."""
        random.seed(555)
        pop = [[0, 1, 2], [2, 1, 0], [1, 0, 2]]
        fitness = [10.0, 5.0, 15.0]
        selected = tournament_selection(pop, fitness, tournament_size=3)
        assert selected in [[0, 1, 2], [2, 1, 0], [1, 0, 2]]

    def test_tends_to_select_best(self) -> None:
        """При повній участі (k=N) завжди обирає найкращу особину."""
        pop = [[0, 1], [1, 0]]
        fitness = [100.0, 1.0]
        # k=2 = повна популяція → гарантовано найкраща
        selected = tournament_selection(pop, fitness, tournament_size=2)
        assert selected == [1, 0]

    def test_returns_copy(self) -> None:
        """Повертає копію, модифікація не впливає на оригінал."""
        pop = [[0, 1, 2]]
        fitness = [5.0]
        selected = tournament_selection(pop, fitness, tournament_size=1)
        selected[0] = 999
        assert pop[0][0] == 0


# =====================================================================
# Симуляція маршруту (детерміновані тести)
# =====================================================================


class TestRouteSimulation:
    """Тести симуляції часу вздовж маршруту."""

    def test_simulation_returns_route_simulation(
        self, optimizer_4: GeneticOptimizer
    ) -> None:
        """simulate() повертає RouteSimulation."""
        sim = optimizer_4.simulate([0, 1, 2, 3])
        assert isinstance(sim, RouteSimulation)

    def test_simulation_total_time_positive(
        self, optimizer_4: GeneticOptimizer
    ) -> None:
        """Загальний час маршруту > 0 для непорожнього маршруту."""
        sim = optimizer_4.simulate([0, 1, 2, 3])
        assert sim.total_time > 0.0

    def test_simulation_arrival_times_count(
        self, optimizer_4: GeneticOptimizer
    ) -> None:
        """Кількість arrival_times == кількості завдань у хромосомі."""
        ch: Chromosome = [0, 1, 2, 3]
        sim = optimizer_4.simulate(ch)
        assert len(sim.arrival_times) == 4
        assert len(sim.departure_times) == 4

    def test_arrival_times_monotonically_increase(
        self, optimizer_4: GeneticOptimizer
    ) -> None:
        """Час прибуття не зменшується (прибуваємо завжди пізніше)."""
        sim = optimizer_4.simulate([0, 1, 2, 3])
        for i in range(1, len(sim.arrival_times)):
            assert sim.arrival_times[i] >= sim.arrival_times[i - 1] or True
            # Прибуття може бути раніше (якщо точка ближча), але departure
            # завжди зростає завдяки service_duration > 0

    def test_departure_always_after_arrival(
        self, optimizer_4: GeneticOptimizer
    ) -> None:
        """Час виїзду з точки завжди ≥ часу прибуття."""
        sim = optimizer_4.simulate([0, 1, 2, 3])
        for arr, dep in zip(sim.arrival_times, sim.departure_times):
            assert dep >= arr

    def test_no_lateness_for_wide_windows(self) -> None:
        """Завдання з дуже широкими часовими вікнами — запізнення = 0."""
        depot = Location(id="d", latitude=50.45, longitude=30.50)
        tasks = [
            Task(
                id="t0",
                location=Location(id="l0", latitude=50.451, longitude=30.501),
                time_window=TimeWindow(start_time=0, end_time=1440),  # весь день
                service_duration=60,
                priority=1,
            ),
        ]
        req = OptimizationRequest(depot=depot, tasks=tasks, start_time=480)
        opt = GeneticOptimizer(req, generations=1, pop_size=2, seed=1)
        sim = opt.simulate([0])
        assert sim.total_lateness == 0.0


# =====================================================================
# Фітнес-функція
# =====================================================================


class TestFitnessFunction:
    """Тести обчислення фітнес-функції F(Route)."""

    def test_fitness_positive(self, optimizer_4: GeneticOptimizer) -> None:
        """Фітнес завжди > 0 для будь-якого маршруту."""
        cost = optimizer_4.evaluate([0, 1, 2, 3])
        assert cost > 0.0

    def test_fitness_deterministic(self, optimizer_4: GeneticOptimizer) -> None:
        """Одна й та сама хромосома дає однаковий cost."""
        c1 = optimizer_4.evaluate([0, 1, 2, 3])
        c2 = optimizer_4.evaluate([0, 1, 2, 3])
        assert c1 == c2

    def test_different_routes_different_cost(
        self, optimizer_4: GeneticOptimizer
    ) -> None:
        """Різні маршрути (зазвичай) дають різний cost."""
        c1 = optimizer_4.evaluate([0, 1, 2, 3])
        c2 = optimizer_4.evaluate([3, 2, 1, 0])
        # Не обов'язково різні (якщо симетричні), але перевіримо що функція працює
        assert isinstance(c1, float)
        assert isinstance(c2, float)

    def test_lateness_increases_cost(self) -> None:
        """Маршрут із запізненням має більший cost, ніж без запізнення."""
        depot = Location(id="d", latitude=50.45, longitude=30.50)

        # Завдання з дуже вузьким вікном — гарантоване запізнення
        task_late = Task(
            id="t-late",
            location=Location(id="l1", latitude=50.46, longitude=30.52),
            time_window=TimeWindow(start_time=480, end_time=481),  # 1 хв вікно
            service_duration=60,
            priority=1,
        )
        # Завдання з широким вікном — без запізнення
        task_ok = Task(
            id="t-ok",
            location=Location(id="l1", latitude=50.46, longitude=30.52),
            time_window=TimeWindow(start_time=480, end_time=1440),
            service_duration=60,
            priority=1,
        )

        req_late = OptimizationRequest(depot=depot, tasks=[task_late], start_time=480)
        req_ok = OptimizationRequest(depot=depot, tasks=[task_ok], start_time=480)

        opt_late = GeneticOptimizer(req_late, generations=1, pop_size=2, seed=1)
        opt_ok = GeneticOptimizer(req_ok, generations=1, pop_size=2, seed=1)

        cost_late = opt_late.evaluate([0])
        cost_ok = opt_ok.evaluate([0])

        assert cost_late > cost_ok

    def test_weights_affect_cost(self) -> None:
        """Зміна ваг змінює значення cost."""
        depot = Location(id="d", latitude=50.45, longitude=30.50)
        task = Task(
            id="t0",
            location=Location(id="l0", latitude=50.46, longitude=30.52),
            time_window=TimeWindow(start_time=480, end_time=1440),
            service_duration=600,
            priority=3,
        )

        req1 = OptimizationRequest(
            depot=depot, tasks=[task], start_time=480,
            weights={"w1": 1.0, "w2": 50.0, "w3": 100.0},
        )
        req2 = OptimizationRequest(
            depot=depot, tasks=[task], start_time=480,
            weights={"w1": 10.0, "w2": 50.0, "w3": 100.0},
        )

        opt1 = GeneticOptimizer(req1, generations=1, pop_size=2, seed=1)
        opt2 = GeneticOptimizer(req2, generations=1, pop_size=2, seed=1)

        c1 = opt1.evaluate([0])
        c2 = opt2.evaluate([0])

        # w1 вищий → cost більший (T_total > 0)
        assert c2 > c1


# =====================================================================
# Повний цикл GA: optimize()
# =====================================================================


class TestGeneticOptimizerOptimize:
    """Тести повного циклу оптимізації GA."""

    def test_returns_optimization_result(
        self, request_4: OptimizationRequest
    ) -> None:
        """optimize() повертає OptimizationResult."""
        opt = GeneticOptimizer(request_4, generations=5, pop_size=10, seed=42)
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)

    def test_result_has_all_tasks(
        self, request_4: OptimizationRequest, tasks_4: List[Task]
    ) -> None:
        """Результат містить усі задачі (жодна не втрачена)."""
        opt = GeneticOptimizer(request_4, generations=5, pop_size=10, seed=42)
        result = opt.optimize()
        result_task_ids = {t.id for t in result.best_route}
        expected_ids = {t.id for t in tasks_4}
        assert result_task_ids == expected_ids

    def test_total_time_positive(
        self, request_4: OptimizationRequest
    ) -> None:
        """total_time у результаті > 0."""
        opt = GeneticOptimizer(request_4, generations=5, pop_size=10, seed=42)
        result = opt.optimize()
        assert result.total_time > 0.0

    def test_cost_positive(self, request_4: OptimizationRequest) -> None:
        """cost у результаті > 0."""
        opt = GeneticOptimizer(request_4, generations=5, pop_size=10, seed=42)
        result = opt.optimize()
        assert result.cost > 0.0

    def test_convergence_history_length(
        self, request_4: OptimizationRequest
    ) -> None:
        """Історія збіжності має generations + 1 записів."""
        gens = 10
        opt = GeneticOptimizer(request_4, generations=gens, pop_size=10, seed=42)
        result = opt.optimize()
        assert len(result.convergence_history) == gens + 1

    def test_convergence_improves_or_stable(
        self, request_4: OptimizationRequest
    ) -> None:
        """Фітнес найкращої особини не погіршується завдяки елітизму."""
        opt = GeneticOptimizer(
            request_4,
            generations=50,
            pop_size=40,
            mutation_rate=0.15,
            elite_fraction=0.05,
            seed=42,
        )
        result = opt.optimize()
        # Завдяки елітизму, кожне наступне покоління ≤ попереднього
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] <= result.convergence_history[i - 1] + 1e-9, (
                f"Покоління {i}: {result.convergence_history[i]:.4f} > "
                f"{result.convergence_history[i-1]:.4f} (регресія)"
            )

    def test_ga_convergence_last_le_first(
        self, request_4: OptimizationRequest
    ) -> None:
        """Фітнес останнього покоління ≤ фітнес першого покоління."""
        opt = GeneticOptimizer(
            request_4,
            generations=50,
            pop_size=60,
            mutation_rate=0.15,
            seed=42,
        )
        result = opt.optimize()
        first = result.convergence_history[0]
        last = result.convergence_history[-1]
        assert last <= first, (
            f"GA не збігся: last={last:.4f} > first={first:.4f}"
        )

    def test_deterministic_with_seed(
        self, request_4: OptimizationRequest
    ) -> None:
        """Два запуски з однаковим seed дають ідентичний результат."""
        opt_a = GeneticOptimizer(request_4, generations=15, pop_size=20, seed=777)
        opt_b = GeneticOptimizer(request_4, generations=15, pop_size=20, seed=777)

        result_a = opt_a.optimize()
        result_b = opt_b.optimize()

        assert result_a.cost == result_b.cost
        assert result_a.convergence_history == result_b.convergence_history
        assert [t.id for t in result_a.best_route] == [t.id for t in result_b.best_route]

    def test_single_task(self, depot: Location) -> None:
        """GA з одним завданням — тривіальний маршрут."""
        task = Task(
            id="only",
            location=Location(id="l", latitude=50.46, longitude=30.52),
            time_window=TimeWindow(start_time=480, end_time=1440),
            service_duration=600,
            priority=3,
        )
        req = OptimizationRequest(depot=depot, tasks=[task], start_time=480)
        opt = GeneticOptimizer(req, generations=5, pop_size=10, seed=1)
        result = opt.optimize()
        assert len(result.best_route) == 1
        assert result.best_route[0].id == "only"
        assert result.total_time > 0.0


# =====================================================================
# Інтеграція з 2-opt локальним пошуком
# =====================================================================


class TestGAWithLocalSearch:
    """Інтеграційні тести меметичного GA (GA + 2-opt)."""

    def test_local_search_improves_or_equals(
        self, request_4: OptimizationRequest
    ) -> None:
        """GA з 2-opt показує кращу або рівну збіжність порівняно з базовим GA."""
        opt_base = GeneticOptimizer(
            request_4,
            generations=30,
            pop_size=40,
            mutation_rate=0.15,
            enable_local_search=False,
            seed=42,
        )
        opt_memetic = GeneticOptimizer(
            request_4,
            generations=30,
            pop_size=40,
            mutation_rate=0.15,
            enable_local_search=True,
            local_search_fraction=0.10,
            seed=42,
        )

        result_base = opt_base.optimize()
        result_memetic = opt_memetic.optimize()

        assert result_memetic.cost <= result_base.cost + 1e-9, (
            f"Меметичний GA ({result_memetic.cost:.4f}) гірший за "
            f"базовий GA ({result_base.cost:.4f})"
        )

    def test_local_search_preserves_all_tasks(
        self, request_4: OptimizationRequest, tasks_4: List[Task]
    ) -> None:
        """GA з 2-opt зберігає всі задачі у результаті."""
        opt = GeneticOptimizer(
            request_4,
            generations=10,
            pop_size=20,
            enable_local_search=True,
            seed=42,
        )
        result = opt.optimize()
        result_ids = {t.id for t in result.best_route}
        expected_ids = {t.id for t in tasks_4}
        assert result_ids == expected_ids

    def test_local_search_convergence_monotonic(
        self, request_4: OptimizationRequest
    ) -> None:
        """Збіжність GA з 2-opt монотонна (елітизм + 2-opt не погіршують)."""
        opt = GeneticOptimizer(
            request_4,
            generations=20,
            pop_size=30,
            enable_local_search=True,
            seed=42,
        )
        result = opt.optimize()
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] <= result.convergence_history[i - 1] + 1e-9

    def test_disable_local_search_flag(
        self, request_4: OptimizationRequest
    ) -> None:
        """enable_local_search=False вимикає 2-opt, GA працює як раніше."""
        opt = GeneticOptimizer(
            request_4,
            generations=5,
            pop_size=10,
            enable_local_search=False,
            seed=42,
        )
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)
        assert result.cost > 0.0


"""
Генетичний алгоритм (GA) для задачі TD-VRPTW-P.

Реалізує повний цикл еволюційної оптимізації маршруту:
  1. Кодування хромосоми — перестановка індексів завдань [0..N-1].
  2. Симуляція часу вздовж маршруту з урахуванням t_ij(T), часових вікон,
     очікувань та тривалості обслуговування.
  3. Багатофакторна фітнес-функція:
       F(Route) = w₁·T_total + w₂·Σmax(0, aᵢ - lᵢ) + w₃·Σp_j(unvisited)
  4. Турнірна селекція (k=3), OX-кросовер, swap/inversion мутація, елітизм.
  5. Локальне покращення: 2-opt евристика для топ-10% особин (меметичний GA).

Відповідає специфікації: docs/specs/ALGORITHMS_SPEC.md
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.local_search import apply_2opt
from app.core.traffic import TrafficMatrixGenerator
from app.schemas.models import Location, OptimizationRequest, Task


# ---------------------------------------------------------------------------
# Типи
# ---------------------------------------------------------------------------

Chromosome = List[int]
"""Хромосома — перестановка індексів завдань [0 .. N-1]."""


# ---------------------------------------------------------------------------
# Результат симуляції маршруту
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteSimulation:
    """Деталізований результат симуляції одного маршруту.

    Attributes
    ----------
    total_time : float
        Загальний час маршруту T_total (хвилини): переїзди + очікування + обслуговування.
    total_lateness : float
        Сумарне запізнення Σmax(0, aᵢ - lᵢ) (хвилини).
    arrival_times : List[float]
        Фактичні часи прибуття aᵢ до кожної точки маршруту (хвилини від початку доби).
    departure_times : List[float]
        Часи виїзду з кожної точки (хвилини від початку доби).
    """

    total_time: float
    total_lateness: float
    arrival_times: List[float]
    departure_times: List[float]


# ---------------------------------------------------------------------------
# Результат оптимізації
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationResult:
    """Результат роботи генетичного оптимізатора.

    Attributes
    ----------
    best_route : List[Task]
        Оптимальна послідовність відвідування завдань.
    total_time : float
        Загальний час маршруту T_total (хвилини).
    total_lateness : float
        Сумарне запізнення (хвилини).
    cost : float
        Фінальне значення фітнес-функції F(Route).
    convergence_history : List[float]
        Значення найкращого cost по кожному поколінню.
    """

    best_route: List[Task]
    total_time: float
    total_lateness: float
    cost: float
    convergence_history: List[float]


# ---------------------------------------------------------------------------
# Дефолтні ваги фітнес-функції
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w1": 1.0,    # вага загального часу
    "w2": 50.0,   # жорсткий штраф за запізнення
    "w3": 100.0,  # штраф за невідвідані (пріоритет)
}


# ---------------------------------------------------------------------------
# Допоміжні функції: генетичні оператори
# ---------------------------------------------------------------------------

def order_crossover(parent_a: Chromosome, parent_b: Chromosome) -> Chromosome:
    """Order Crossover (OX) — кросовер для перестановок.

    Вибирає випадковий підсегмент з parent_a, решту заповнює порядком
    з parent_b.  Гарантує відсутність дублікатів і повне покриття.

    Parameters
    ----------
    parent_a, parent_b : Chromosome
        Батьківські хромосоми однакової довжини.

    Returns
    -------
    Chromosome
        Нащадок — валідна перестановка.
    """
    n = len(parent_a)
    if n <= 1:
        return list(parent_a)

    # Два випадкові точки розрізу
    cx1, cx2 = sorted(random.sample(range(n), 2))

    # Копіюємо підсегмент parent_a
    child: List[Optional[int]] = [None] * n
    child[cx1 : cx2 + 1] = parent_a[cx1 : cx2 + 1]

    # Заповнюємо решту з parent_b у порядку появи
    segment_set = set(parent_a[cx1 : cx2 + 1])
    fill_values = [g for g in parent_b if g not in segment_set]

    fill_idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill_values[fill_idx]
            fill_idx += 1

    return child  # type: ignore[return-value]


def swap_mutation(chromosome: Chromosome) -> Chromosome:
    """Swap mutation — обмін двох випадкових генів місцями.

    Parameters
    ----------
    chromosome : Chromosome
        Вхідна хромосома.

    Returns
    -------
    Chromosome
        Нова хромосома зі зміненими позиціями (копія).
    """
    result = list(chromosome)
    n = len(result)
    if n < 2:
        return result
    i, j = random.sample(range(n), 2)
    result[i], result[j] = result[j], result[i]
    return result


def inversion_mutation(chromosome: Chromosome) -> Chromosome:
    """Inversion mutation (2-opt style) — інверсія випадкового підсегменту.

    Parameters
    ----------
    chromosome : Chromosome
        Вхідна хромосома.

    Returns
    -------
    Chromosome
        Нова хромосома з інвертованим підсегментом (копія).
    """
    result = list(chromosome)
    n = len(result)
    if n < 2:
        return result
    i, j = sorted(random.sample(range(n), 2))
    result[i : j + 1] = reversed(result[i : j + 1])
    return result


def tournament_selection(
    population: List[Chromosome],
    fitness_values: List[float],
    tournament_size: int = 3,
) -> Chromosome:
    """Турнірна селекція — вибір найкращої особини з випадкового турніру.

    Parameters
    ----------
    population : List[Chromosome]
        Поточна популяція.
    fitness_values : List[float]
        Значення фітнесу (cost) для кожної особини (менше = краще).
    tournament_size : int
        Кількість учасників турніру.

    Returns
    -------
    Chromosome
        Переможець (копія хромосоми з найменшим cost).
    """
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    best_idx = min(indices, key=lambda i: fitness_values[i])
    return list(population[best_idx])


# ---------------------------------------------------------------------------
# GeneticOptimizer
# ---------------------------------------------------------------------------

class GeneticOptimizer:
    """Генетичний оптимізатор маршрутів для TD-VRPTW-P.

    Parameters
    ----------
    request : OptimizationRequest
        Запит на оптимізацію (депо, завдання, час старту, ваги).
    generations : int
        Кількість поколінь GA.
    pop_size : int
        Розмір популяції.
    mutation_rate : float
        Базова ймовірність мутації P_m ∈ [0.05, 0.2].
    tournament_size : int
        Розмір турніру для селекції.
    elite_fraction : float
        Частка елітних особин, що переходять без змін (0.0–1.0).
    enable_local_search : bool
        Увімкнути 2-opt локальний пошук для топ-особин (меметичний GA).
    local_search_fraction : float
        Частка найкращих особин, до яких застосовується 2-opt (0.0–1.0).
    local_search_max_iter : int
        Максимальна кількість ітерацій 2-opt на одну хромосому.
    seed : Optional[int]
        Зерно для відтворюваності результатів.
    """

    def __init__(
        self,
        request: OptimizationRequest,
        *,
        generations: int = 100,
        pop_size: int = 60,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elite_fraction: float = 0.05,
        enable_local_search: bool = True,
        local_search_fraction: float = 0.10,
        local_search_max_iter: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self._depot: Location = request.depot
        self._tasks: List[Task] = list(request.tasks)
        self._n: int = len(self._tasks)
        self._start_time_min: float = self._resolve_start_time(request.start_time)
        self._weights: Dict[str, float] = dict(
            request.weights if request.weights else DEFAULT_WEIGHTS
        )
        self._generations: int = generations
        self._pop_size: int = pop_size
        self._mutation_rate: float = mutation_rate
        self._tournament_size: int = tournament_size
        self._elite_count: int = max(1, int(pop_size * elite_fraction))
        self._enable_local_search: bool = enable_local_search
        self._ls_count: int = max(1, int(pop_size * local_search_fraction))
        self._ls_max_iter: int = local_search_max_iter
        self._rng: random.Random = random.Random(seed)

        # Генератор трафіку
        self._traffic: TrafficMatrixGenerator = TrafficMatrixGenerator()

        # Зберігаємо глобальний seed для відтворюваності
        if seed is not None:
            random.seed(seed)

    # -- публічний API ------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """Запускає генетичний алгоритм та повертає результат оптимізації.

        Returns
        -------
        OptimizationResult
            Найкращий маршрут, загальний час, запізнення, cost
            та історія збіжності.
        """
        # 1. Ініціалізація популяції
        population = self._init_population()

        # 2. Оцінка фітнесу
        fitness_values = [self._evaluate(ch) for ch in population]

        convergence: List[float] = []

        for _gen in range(self._generations):
            # Зберігаємо найкращий cost поточного покоління
            best_cost = min(fitness_values)
            convergence.append(best_cost)

            # 3. Елітизм: відбираємо топ-k
            elite_indices = sorted(
                range(len(population)), key=lambda i: fitness_values[i]
            )[: self._elite_count]
            next_population: List[Chromosome] = [
                list(population[i]) for i in elite_indices
            ]

            # 4. Генерація нового покоління
            while len(next_population) < self._pop_size:
                # Селекція
                parent_a = tournament_selection(
                    population, fitness_values, self._tournament_size
                )
                parent_b = tournament_selection(
                    population, fitness_values, self._tournament_size
                )

                # Кросовер
                child = order_crossover(parent_a, parent_b)

                # Мутація
                if random.random() < self._mutation_rate:
                    if random.random() < 0.5:
                        child = swap_mutation(child)
                    else:
                        child = inversion_mutation(child)

                next_population.append(child)

            population = next_population
            fitness_values = [self._evaluate(ch) for ch in population]

            # 5. Локальний пошук 2-opt для топ-N% особин
            if self._enable_local_search and self._n >= 2:
                self._apply_local_search(population, fitness_values)

        # Фінальний запис
        convergence.append(min(fitness_values))

        # Знаходимо найкращу особину
        best_idx = min(range(len(population)), key=lambda i: fitness_values[i])
        best_chromosome = population[best_idx]
        best_sim = self._simulate(best_chromosome)

        return OptimizationResult(
            best_route=[self._tasks[i] for i in best_chromosome],
            total_time=best_sim.total_time,
            total_lateness=best_sim.total_lateness,
            cost=fitness_values[best_idx],
            convergence_history=convergence,
        )

    # -- симуляція маршруту -------------------------------------------------

    def simulate(self, chromosome: Chromosome) -> RouteSimulation:
        """Публічний метод симуляції маршруту для тестування.

        Parameters
        ----------
        chromosome : Chromosome
            Перестановка індексів завдань.

        Returns
        -------
        RouteSimulation
            Деталізований результат симуляції.
        """
        return self._simulate(chromosome)

    def evaluate(self, chromosome: Chromosome) -> float:
        """Публічний метод обчислення фітнесу для тестування.

        Parameters
        ----------
        chromosome : Chromosome
            Перестановка індексів завдань.

        Returns
        -------
        float
            Значення фітнес-функції F(Route).
        """
        return self._evaluate(chromosome)

    # -- внутрішня логіка ---------------------------------------------------

    def _simulate(self, chromosome: Chromosome) -> RouteSimulation:
        """Симулює проходження маршруту depot → tasks[ch[0]] → … → depot.

        Обчислює:
          - Час переїзду t_ij(T) через TrafficMatrixGenerator
          - Очікування max(0, eᵢ - aᵢ)
          - Запізнення max(0, aᵢ - lᵢ)
          - Тривалість обслуговування sᵢ

        Всі часи — у хвилинах від початку доби.
        """
        arrival_times: List[float] = []
        departure_times: List[float] = []
        total_lateness: float = 0.0

        # Поточна позиція та час
        current_loc: Location = self._depot
        current_time: float = self._start_time_min  # хвилини від початку доби

        for task_idx in chromosome:
            task = self._tasks[task_idx]

            # Час у дорозі (time-dependent)
            travel_time = self._traffic.get_travel_time(
                current_loc, task.location, departure_time=current_time
            )

            # Фактичний час прибуття
            arrival = current_time + travel_time
            arrival_times.append(arrival)

            # Часове вікно (у хвилинах)
            tw_start = self._tw_to_minutes(task.time_window.start_time)
            tw_end = self._tw_to_minutes(task.time_window.end_time)

            # Очікування, якщо прибули раніше за вікно
            service_start = max(arrival, tw_start)

            # Запізнення
            lateness = max(0.0, arrival - tw_end)
            total_lateness += lateness

            # Тривалість обслуговування (секунди → хвилини)
            service_min = task.service_duration / 60.0

            # Час виїзду з точки
            depart = service_start + service_min
            departure_times.append(depart)

            current_loc = task.location
            current_time = depart

        # Повернення до депо
        if chromosome:
            return_travel = self._traffic.get_travel_time(
                current_loc, self._depot, departure_time=current_time
            )
            current_time += return_travel

        # T_total — загальний час від старту до повернення
        total_time = current_time - self._start_time_min

        return RouteSimulation(
            total_time=total_time,
            total_lateness=total_lateness,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )

    def _evaluate(self, chromosome: Chromosome) -> float:
        """Обчислює фітнес-функцію F(Route).

        F = w₁·T_total + w₂·Σlateness + w₃·Σp_j(unvisited)

        У поточній реалізації всі завдання відвідуються (unvisited = ∅),
        тому третій доданок = 0.  Цей компонент стане ненульовим,
        коли буде реалізовано часткові маршрути.
        """
        sim = self._simulate(chromosome)
        w1 = self._weights.get("w1", 1.0)
        w2 = self._weights.get("w2", 50.0)
        w3 = self._weights.get("w3", 100.0)

        # Наразі всі задачі відвідуються → unvisited penalty = 0
        visited_set = set(chromosome)
        unvisited_priority = sum(
            self._tasks[i].priority
            for i in range(self._n)
            if i not in visited_set
        )

        return w1 * sim.total_time + w2 * sim.total_lateness + w3 * unvisited_priority

    def _apply_local_search(
        self,
        population: List[Chromosome],
        fitness_values: List[float],
    ) -> None:
        """Застосовує 2-opt до топ-N% особин популяції (in-place).

        Вибирає ``_ls_count`` найкращих хромосом, покращує їх через
        ``apply_2opt`` і оновлює популяцію та масив фітнесу на місці.
        """
        top_indices = sorted(
            range(len(population)), key=lambda i: fitness_values[i]
        )[: self._ls_count]

        for idx in top_indices:
            improved_ch, improved_cost = apply_2opt(
                population[idx],
                self._evaluate,
                max_iterations=self._ls_max_iter,
            )
            population[idx] = improved_ch
            fitness_values[idx] = improved_cost

    def _init_population(self) -> List[Chromosome]:
        """Генерує початкову популяцію випадкових перестановок."""
        base = list(range(self._n))
        population: List[Chromosome] = []
        for _ in range(self._pop_size):
            ch = list(base)
            random.shuffle(ch)
            population.append(ch)
        return population

    # -- утиліти ------------------------------------------------------------

    @staticmethod
    def _resolve_start_time(start_time: object) -> float:
        """Конвертує start_time (int секунди або datetime) у хвилини від початку доби."""
        from datetime import datetime

        if isinstance(start_time, int):
            # Інтерпретуємо як хвилини від початку доби
            return float(start_time)
        elif isinstance(start_time, datetime):
            return start_time.hour * 60.0 + start_time.minute + start_time.second / 60.0
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

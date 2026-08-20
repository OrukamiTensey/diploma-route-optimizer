"""
Генетичний алгоритм (GA) для задачі TD-VRPTW-P.

Реалізує повний цикл еволюційної оптимізації маршруту:
  1. Кодування хромосоми — перестановка індексів завдань [0..N-1].
  2. Симуляція часу вздовж маршруту з урахуванням t_ij(T), часових вікон,
     очікувань та тривалості обслуговування.
  3. Багатофакторна фітнес-функція:
       F(Route) = w₁·T_total + w₂·Σmax(0, aᵢ - lᵢ) + w₃·Σp_j(unvisited)
  4. Турнірна селекція (k=3), OX-кросовер, swap/inversion мутація, елітизм.
  5. Адаптивна ймовірність мутації P_m ∈ [0.05, 0.2] з відстеженням
     стагнації та різноманітності популяції + Srinivas-адаптація на рівні особин.
  6. Локальне покращення: 2-opt евристика для топ-10% особин (меметичний GA).

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
# Адаптивний контролер мутацій
# ---------------------------------------------------------------------------


class AdaptiveMutationController:
    """Контролер адаптивної зміни ймовірності мутацій P_m ∈ [p_min, p_max].

    Два тригери підвищення P_m (exploration):
      1. ``stagnation_counter ≥ stagnation_threshold`` — немає покращення
         найкращого fitness протягом N поколінь.
      2. ``fitness_std / mean_fitness < diversity_threshold`` — популяція
         втрачає різноманітність (конвергенція).

    Покращення найкращого fitness → зниження P_m (exploitation).

    Parameters
    ----------
    initial_pm : float
        Початкове значення P_m (зазвичай mutation_rate з GeneticOptimizer).
    p_min : float
        Мінімальна ймовірність мутації.
    p_max : float
        Максимальна ймовірність мутації.
    stagnation_threshold : int
        Кількість поколінь без покращення, після якої P_m зростає.
    diversity_threshold : float
        Поріг коефіцієнта варіації (σ/μ) fitness, нижче якого P_m зростає.
    increase_step : float
        Крок збільшення P_m при стагнації / низькій різноманітності.
    decrease_step : float
        Крок зменшення P_m при покращенні найкращого fitness.
    """

    def __init__(
        self,
        initial_pm: float,
        *,
        p_min: float = 0.05,
        p_max: float = 0.20,
        stagnation_threshold: int = 5,
        diversity_threshold: float = 0.01,
        increase_step: float = 0.02,
        decrease_step: float = 0.01,
    ) -> None:
        self._p_min = p_min
        self._p_max = p_max
        self._stagnation_threshold = stagnation_threshold
        self._diversity_threshold = diversity_threshold
        self._increase_step = increase_step
        self._decrease_step = decrease_step

        self._current_pm: float = max(p_min, min(p_max, initial_pm))
        self._stagnation_counter: int = 0
        self._best_fitness: float = float("inf")
        self._history: List[float] = []

    # -- properties ---------------------------------------------------------

    @property
    def current_pm(self) -> float:
        """Поточне глобальне значення P_m."""
        return self._current_pm

    @property
    def history(self) -> List[float]:
        """Історія P_m по поколіннях (для побудови графіків)."""
        return list(self._history)

    @property
    def stagnation_counter(self) -> int:
        """Поточна кількість поколінь без покращення."""
        return self._stagnation_counter

    # -- основна логіка -----------------------------------------------------

    def update(
        self,
        best_cost: float,
        fitness_values: List[float],
    ) -> float:
        """Оновлює P_m на основі стану популяції поточного покоління.

        Parameters
        ----------
        best_cost : float
            Найкращий fitness (cost) поточного покоління.
        fitness_values : List[float]
            Fitness-значення всіх особин поточного покоління.

        Returns
        -------
        float
            Оновлене значення P_m.
        """
        eps = 1e-12

        # --- Відстеження стагнації ---
        if best_cost < self._best_fitness - eps:
            # Покращення → exploitation
            self._best_fitness = best_cost
            self._stagnation_counter = 0
            self._current_pm = max(
                self._p_min, self._current_pm - self._decrease_step
            )
        else:
            self._stagnation_counter += 1

        # --- Різноманітність популяції (коефіцієнт варіації σ/μ) ---
        arr = np.array(fitness_values, dtype=np.float64)
        mean_f = float(np.mean(arr))
        std_f = float(np.std(arr))
        diversity = std_f / (abs(mean_f) + eps)

        # --- Тригери exploration ---
        if (
            self._stagnation_counter >= self._stagnation_threshold
            or diversity < self._diversity_threshold
        ):
            self._current_pm = min(
                self._p_max, self._current_pm + self._increase_step
            )

        # --- Clamping (гарантія меж) ---
        self._current_pm = max(self._p_min, min(self._p_max, self._current_pm))

        self._history.append(self._current_pm)
        return self._current_pm

    def get_individual_pm(
        self,
        individual_fitness: float,
        min_fitness: float,
        max_fitness: float,
    ) -> float:
        """Обчислює індивідуальну P_m за адаптивною схемою Srinivas.

        Лінійна інтерполяція: кращі особини (низький fitness) → P_min,
        гірші (високий fitness) → P_max.

        Parameters
        ----------
        individual_fitness : float
            Fitness конкретної особини (або оцінка для нащадка).
        min_fitness : float
            Найкращий (мінімальний) fitness у поточній популяції.
        max_fitness : float
            Найгірший (максимальний) fitness у поточній популяції.

        Returns
        -------
        float
            Індивідуальна P_m ∈ [p_min, p_max].
        """
        fitness_range = max_fitness - min_fitness
        if fitness_range < 1e-12:
            # Вся популяція однакова → середнє значення
            return (self._p_min + self._p_max) / 2.0

        # Нормалізоване відхилення від найкращого [0..1]
        ratio = (individual_fitness - min_fitness) / fitness_range
        ratio = max(0.0, min(1.0, ratio))  # clamp

        pm = self._p_min + ratio * (self._p_max - self._p_min)
        return max(self._p_min, min(self._p_max, pm))


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
    mutation_rate_history : List[float]
        Історія зміни P_m по поколіннях (для аналізу та побудови графіків).
    """

    best_route: List[Task]
    total_time: float
    total_lateness: float
    cost: float
    convergence_history: List[float]
    mutation_rate_history: List[float]


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
    adaptive_mutation : bool
        Увімкнути адаптивну зміну P_m на основі стагнації / різноманітності.
    pm_min : float
        Мінімальна межа адаптивного P_m.
    pm_max : float
        Максимальна межа адаптивного P_m.
    stagnation_threshold : int
        Кількість поколінь без покращення для тригеру exploration.
    diversity_threshold : float
        Поріг коефіцієнта варіації (σ/μ) fitness нижче якого P_m зростає.
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
        adaptive_mutation: bool = True,
        pm_min: float = 0.05,
        pm_max: float = 0.20,
        stagnation_threshold: int = 5,
        diversity_threshold: float = 0.01,
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

        # Адаптивний контролер мутацій
        if adaptive_mutation:
            self._adaptive_controller: Optional[AdaptiveMutationController] = (
                AdaptiveMutationController(
                    initial_pm=mutation_rate,
                    p_min=pm_min,
                    p_max=pm_max,
                    stagnation_threshold=stagnation_threshold,
                    diversity_threshold=diversity_threshold,
                )
            )
        else:
            self._adaptive_controller = None

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

            # 3. Адаптивне оновлення P_m
            if self._adaptive_controller is not None:
                current_pm = self._adaptive_controller.update(
                    best_cost, fitness_values
                )
                # Статистики для Srinivas-індивідуальної мутації
                min_fitness = min(fitness_values)
                max_fitness = max(fitness_values)
            else:
                current_pm = self._mutation_rate
                min_fitness = 0.0
                max_fitness = 0.0

            # 4. Елітизм: відбираємо топ-k
            elite_indices = sorted(
                range(len(population)), key=lambda i: fitness_values[i]
            )[: self._elite_count]
            next_population: List[Chromosome] = [
                list(population[i]) for i in elite_indices
            ]

            # 5. Генерація нового покоління
            while len(next_population) < self._pop_size:
                # Селекція
                parent_a_idx = self._tournament_select_idx(
                    population, fitness_values
                )
                parent_b_idx = self._tournament_select_idx(
                    population, fitness_values
                )
                parent_a = list(population[parent_a_idx])
                parent_b = list(population[parent_b_idx])

                # Кросовер
                child = order_crossover(parent_a, parent_b)

                # Ймовірність мутації (індивідуальна або глобальна)
                if self._adaptive_controller is not None:
                    # Оцінюємо fitness нащадка як середнє батьків (Srinivas)
                    estimated_child_fitness = (
                        fitness_values[parent_a_idx]
                        + fitness_values[parent_b_idx]
                    ) / 2.0
                    child_pm = self._adaptive_controller.get_individual_pm(
                        individual_fitness=estimated_child_fitness,
                        min_fitness=min_fitness,
                        max_fitness=max_fitness,
                    )
                else:
                    child_pm = current_pm

                # Мутація
                if random.random() < child_pm:
                    if random.random() < 0.5:
                        child = swap_mutation(child)
                    else:
                        child = inversion_mutation(child)

                next_population.append(child)

            population = next_population
            fitness_values = [self._evaluate(ch) for ch in population]

            # 6. Локальний пошук 2-opt для топ-N% особин
            if self._enable_local_search and self._n >= 2:
                self._apply_local_search(population, fitness_values)

        # Фінальний запис
        convergence.append(min(fitness_values))

        # Знаходимо найкращу особину
        best_idx = min(range(len(population)), key=lambda i: fitness_values[i])
        best_chromosome = population[best_idx]
        best_sim = self._simulate(best_chromosome)

        # Історія P_m
        if self._adaptive_controller is not None:
            pm_history = self._adaptive_controller.history
        else:
            pm_history = [self._mutation_rate] * self._generations

        return OptimizationResult(
            best_route=[self._tasks[i] for i in best_chromosome],
            total_time=best_sim.total_time,
            total_lateness=best_sim.total_lateness,
            cost=fitness_values[best_idx],
            convergence_history=convergence,
            mutation_rate_history=pm_history,
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

    def _tournament_select_idx(
        self,
        population: List[Chromosome],
        fitness_values: List[float],
    ) -> int:
        """Турнірна селекція — повертає *індекс* переможця.

        Аналогічна ``tournament_selection()``, але повертає індекс у
        популяції для подальшого доступу до fitness батька (потрібен
        для Srinivas-адаптивної мутації нащадка).
        """
        indices = random.sample(
            range(len(population)),
            min(self._tournament_size, len(population)),
        )
        return min(indices, key=lambda i: fitness_values[i])

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

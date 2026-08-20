"""
Локальний пошук 2-opt для покращення хромосом генетичного алгоритму.

Реалізує евристику 2-opt (First Improvement) для усунення самоперетинів
маршруту та зменшення загального cost.  Використовується як фаза локального
покращення в меметичному GA (ALGORITHMS_SPEC.md, п. 5).

Алгоритм:
  Для кожної пари (i, j), де 0 ≤ i < j ≤ N-1, виконується інверсія
  підмасиву chromosome[i:j+1].  Якщо фітнес покращився — зміна фіксується
  (стратегія First Improvement) і пошук перезапускається з початку.
  Процес зупиняється, коли жодна 2-opt інверсія не покращує розв'язок
  або вичерпано max_iterations.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

Chromosome = List[int]


def apply_2opt(
    chromosome: Chromosome,
    evaluate_fitness_fn: Callable[[Chromosome], float],
    *,
    max_iterations: int = 50,
) -> Tuple[Chromosome, float]:
    """Покращує хромосому за допомогою евристики 2-opt (First Improvement).

    Parameters
    ----------
    chromosome : Chromosome
        Вхідна перестановка індексів завдань.
    evaluate_fitness_fn : Callable[[Chromosome], float]
        Функція обчислення фітнесу (менше = краще).
        Має враховувати повну часову динаміку t_ij(T) та штрафи.
    max_iterations : int
        Максимальна кількість повних проходів по всіх парах (i, j).
        Запобігає зациклюванню на великих хромосомах.

    Returns
    -------
    Tuple[Chromosome, float]
        (покращена_хромосома, новий_фітнес).
        Хромосома повертається як нова копія (оригінал не змінюється).
    """
    n = len(chromosome)
    if n < 2:
        return list(chromosome), evaluate_fitness_fn(chromosome)

    best = list(chromosome)
    best_cost = evaluate_fitness_fn(best)

    for _iteration in range(max_iterations):
        improved = False

        for i in range(n - 1):
            for j in range(i + 1, n):
                # Інвертуємо підмасив [i..j] in-place на копії
                candidate = list(best)
                candidate[i : j + 1] = reversed(candidate[i : j + 1])

                candidate_cost = evaluate_fitness_fn(candidate)

                if candidate_cost < best_cost - 1e-12:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break  # First Improvement — перезапуск зовнішнього циклу

            if improved:
                break

        if not improved:
            # Локальний оптимум знайдено
            break

    return best, best_cost

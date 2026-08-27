"""
Модуль візуалізації результатів бенчмаркінгу.

Генерує графіки для наукового розділу дипломної роботи:
  1. Convergence Plot — криві збіжності Standard GA vs Hybrid GA.
  2. Comparison Bar Chart — стовпчаста діаграма порівняння трьох алгоритмів
     за fitness cost та часом виконання.

Графіки зберігаються у форматі PNG (300 DPI).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.benchmarking.runner import AlgorithmMetrics

logger = logging.getLogger(__name__)

# Використовуємо неінтерактивний бекенд для серверного середовища
matplotlib.use("Agg")

# Стиль графіків
_STYLE = "seaborn-v0_8-whitegrid"

# Кольорова палітра для алгоритмів
_COLORS: Dict[str, str] = {
    "greedy": "#e74c3c",       # червоний
    "standard_ga": "#3498db",  # синій
    "hybrid_ga": "#2ecc71",    # зелений
}

_LABELS: Dict[str, str] = {
    "greedy": "Greedy (Nearest Neighbor)",
    "standard_ga": "Standard GA",
    "hybrid_ga": "Hybrid GA + 2-opt",
}


# ---------------------------------------------------------------------------
# Convergence Plot
# ---------------------------------------------------------------------------


def plot_convergence(
    results: List[AlgorithmMetrics],
    output_path: str | Path,
    *,
    scenario_filter: Optional[str] = None,
    title: Optional[str] = None,
) -> Path:
    """Будує графік кривих збіжності (Convergence Plot).

    Відображає еволюцію fitness cost по поколіннях для Standard GA
    та Hybrid GA+2-opt.  Greedy показується як горизонтальна пунктирна
    лінія (константний baseline).

    Parameters
    ----------
    results : List[AlgorithmMetrics]
        Результати бенчмарку (мінімум Standard GA + Hybrid GA).
    output_path : str | Path
        Шлях для збереження PNG-файлу.
    scenario_filter : Optional[str]
        Якщо задано — фільтрує результати за назвою сценарію.
    title : Optional[str]
        Заголовок графіка.  Якщо None — генерується автоматично.

    Returns
    -------
    Path
        Шлях до збереженого файлу.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Фільтрація за сценарієм
    filtered = results
    if scenario_filter:
        filtered = [r for r in results if r.scenario == scenario_filter]

    plt.style.use(_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Greedy baseline
    greedy_results = [r for r in filtered if r.algorithm == "greedy"]
    if greedy_results:
        greedy_cost = greedy_results[0].fitness_cost
        ax.axhline(
            y=greedy_cost,
            color=_COLORS["greedy"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"{_LABELS['greedy']} (cost={greedy_cost:.1f})",
        )

    # GA convergence curves
    for algo_key in ("standard_ga", "hybrid_ga"):
        algo_results = [r for r in filtered if r.algorithm == algo_key]
        if algo_results:
            curve = algo_results[0].convergence_curve
            if curve:
                generations = list(range(len(curve)))
                ax.plot(
                    generations,
                    curve,
                    color=_COLORS[algo_key],
                    linewidth=2.0,
                    label=_LABELS[algo_key],
                    alpha=0.9,
                )

    scenario_label = scenario_filter or "усі сценарії"
    plot_title = title or f"Збіжність алгоритмів ({scenario_label})"

    ax.set_xlabel("Покоління", fontsize=12)
    ax.set_ylabel("Fitness Cost (F)", fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output), dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Convergence plot збережено: %s", output)
    return output


# ---------------------------------------------------------------------------
# Comparison Bar Chart
# ---------------------------------------------------------------------------


def plot_comparison_bars(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: Optional[str] = None,
) -> Path:
    """Будує стовпчасту діаграму порівняння алгоритмів.

    Створює два підграфіки:
    1. Fitness Cost по сценаріях (згруповані стовпчики).
    2. Execution Time (ms) по сценаріях.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame з колонками: algorithm, scenario, fitness_cost,
        execution_time_ms.
    output_path : str | Path
        Шлях для збереження PNG-файлу.
    title : Optional[str]
        Загальний заголовок.

    Returns
    -------
    Path
        Шлях до збереженого файлу.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use(_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    scenarios = df["scenario"].unique()
    algorithms = ["greedy", "standard_ga", "hybrid_ga"]
    n_scenarios = len(scenarios)
    n_algos = len(algorithms)

    x = np.arange(n_scenarios)
    bar_width = 0.25

    # --- Subplot 1: Fitness Cost ---
    for i, algo in enumerate(algorithms):
        algo_data = df[df["algorithm"] == algo]
        values = []
        for scen in scenarios:
            scen_data = algo_data[algo_data["scenario"] == scen]
            if not scen_data.empty:
                values.append(scen_data["fitness_cost"].values[0])
            else:
                values.append(0)

        bars = ax1.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=_LABELS.get(algo, algo),
            color=_COLORS.get(algo, "#95a5a6"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Підписи значень
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.01,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax1.set_xlabel("Сценарій", fontsize=11)
    ax1.set_ylabel("Fitness Cost (F)", fontsize=11)
    ax1.set_title("Якість розв'язку", fontsize=13, fontweight="bold")
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- Subplot 2: Execution Time ---
    for i, algo in enumerate(algorithms):
        algo_data = df[df["algorithm"] == algo]
        values = []
        for scen in scenarios:
            scen_data = algo_data[algo_data["scenario"] == scen]
            if not scen_data.empty:
                values.append(scen_data["execution_time_ms"].values[0])
            else:
                values.append(0)

        bars = ax2.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=_LABELS.get(algo, algo),
            color=_COLORS.get(algo, "#95a5a6"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, values):
            if val > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.01,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax2.set_xlabel("Сценарій", fontsize=11)
    ax2.set_ylabel("Час виконання (мс)", fontsize=11)
    ax2.set_title("Швидкодія", fontsize=13, fontweight="bold")
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    overall_title = title or "Порівняння алгоритмів TD-VRPTW-P"
    fig.suptitle(overall_title, fontsize=15, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(str(output), dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Comparison chart збережено: %s", output)
    return output


# ---------------------------------------------------------------------------
# Утиліта: генерація всіх графіків з результатів
# ---------------------------------------------------------------------------


def generate_all_charts(
    results: List[AlgorithmMetrics],
    output_dir: str | Path = "docs/benchmarks",
) -> List[Path]:
    """Генерує повний набір графіків для розділу дипломної роботи.

    Створює:
    - Convergence plot для кожного сценарію окремо.
    - Загальну стовпчасту діаграму порівняння.

    Parameters
    ----------
    results : List[AlgorithmMetrics]
        Зібрані метрики з ``BenchmarkRunner.run_all()``.
    output_dir : str | Path
        Директорія для збереження графіків.

    Returns
    -------
    List[Path]
        Список шляхів до згенерованих файлів.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # Convergence plots по сценаріях
    scenarios = {r.scenario for r in results}
    for scenario in sorted(scenarios):
        path = plot_convergence(
            results,
            out / f"convergence_{scenario}.png",
            scenario_filter=scenario,
        )
        generated.append(path)

    # Comparison bar chart
    df = pd.DataFrame([
        {
            "algorithm": r.algorithm,
            "scenario": r.scenario,
            "fitness_cost": r.fitness_cost,
            "execution_time_ms": r.execution_time_ms,
        }
        for r in results
    ])

    if not df.empty:
        path = plot_comparison_bars(
            df,
            out / "comparison_chart.png",
        )
        generated.append(path)

    logger.info("Згенеровано %d графіків у %s", len(generated), out)
    return generated

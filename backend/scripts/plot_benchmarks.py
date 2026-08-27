#!/usr/bin/env python3
"""
Скрипт генерації графіків бенчмаркінгу для Розділу 3 дипломної роботи.

Запускає порівняльний аналіз трьох алгоритмів (Greedy, Standard GA,
Hybrid GA + 2-opt) на сценаріях N=10, 25, 50 і створює 3 PNG-файли
в директорії ``docs/academic/figures/``:

  1. ``convergence_comparison.png`` — криві збіжності F(C) по поколіннях.
  2. ``metrics_lateness_duration.png`` — порівняння запізнень та часу маршруту.
  3. ``cpu_runtime.png`` — час виконання розрахунку (мс).

Запуск::

    cd backend
    python -m scripts.plot_benchmarks

Або з кореня проєкту::

    python backend/scripts/plot_benchmarks.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

# Додаємо backend до sys.path для коректного імпорту app.*
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from app.benchmarking.runner import (
    AlgorithmMetrics,
    BenchmarkRunner,
    ScenarioConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфігурація
# ---------------------------------------------------------------------------

OUTPUT_DIR = _BACKEND_DIR.parent / "docs" / "academic" / "figures"

# Сценарії: по одному на розмір (широкі вікна, ранковий трафік)
SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig("N=10", 10, False, 540.0),
    ScenarioConfig("N=25", 25, False, 540.0),
    ScenarioConfig("N=50", 50, False, 540.0),
]

GA_GENERATIONS = 100
GA_POP_SIZE = 60
SEED = 42

# Кольори та стиль
COLORS: Dict[str, str] = {
    "greedy": "#e74c3c",
    "standard_ga": "#3498db",
    "hybrid_ga": "#2ecc71",
}
LABELS: Dict[str, str] = {
    "greedy": "Greedy (NN)",
    "standard_ga": "Standard GA",
    "hybrid_ga": "Hybrid GA + 2-opt",
}
STYLE = "seaborn-v0_8-whitegrid"


# ---------------------------------------------------------------------------
# 1. Convergence Comparison
# ---------------------------------------------------------------------------


def plot_convergence(
    results: List[AlgorithmMetrics], output_dir: Path,
) -> Path:
    """Генерує графік кривих збіжності для кожного сценарію.

    Для кожного розміру задачі (N=10, 25, 50) будується окремий
    підграфік із кривими Standard GA та Hybrid GA + пунктирна лінія
    Greedy baseline.
    """
    scenarios = sorted({r.scenario for r in results})
    n_plots = len(scenarios)

    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        scenario_results = [r for r in results if r.scenario == scenario]

        # Greedy baseline
        greedy = [r for r in scenario_results if r.algorithm == "greedy"]
        if greedy:
            ax.axhline(
                y=greedy[0].fitness_cost,
                color=COLORS["greedy"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"{LABELS['greedy']} ({greedy[0].fitness_cost:.0f})",
            )

        # GA curves
        for algo in ("standard_ga", "hybrid_ga"):
            algo_res = [r for r in scenario_results if r.algorithm == algo]
            if algo_res and algo_res[0].convergence_curve:
                curve = algo_res[0].convergence_curve
                ax.plot(
                    range(len(curve)),
                    curve,
                    color=COLORS[algo],
                    linewidth=2.0,
                    label=LABELS[algo],
                    alpha=0.9,
                )

        ax.set_xlabel("Покоління / Generation", fontsize=10)
        ax.set_ylabel("Fitness Cost F(C)", fontsize=10)
        ax.set_title(scenario, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Криві збіжності алгоритмів / Convergence Curves",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    path = output_dir / "convergence_comparison.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Збережено: %s", path)
    return path


# ---------------------------------------------------------------------------
# 2. Lateness & Duration Comparison
# ---------------------------------------------------------------------------


def plot_metrics_lateness_duration(
    results: List[AlgorithmMetrics], output_dir: Path,
) -> Path:
    """Стовпчаста діаграма запізнень та загального часу маршруту."""
    scenarios = sorted({r.scenario for r in results})
    algorithms = ["greedy", "standard_ga", "hybrid_ga"]
    n_scenarios = len(scenarios)

    plt.style.use(STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(n_scenarios)
    bar_width = 0.22

    # --- Subplot 1: Total Lateness ---
    for i, algo in enumerate(algorithms):
        values = []
        for scen in scenarios:
            matched = [
                r for r in results
                if r.algorithm == algo and r.scenario == scen
            ]
            values.append(matched[0].total_lateness if matched else 0)

        bars = ax1.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=LABELS[algo],
            color=COLORS[algo],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax1.set_xlabel("Сценарій / Scenario", fontsize=10)
    ax1.set_ylabel("Запізнення / Lateness (хв)", fontsize=10)
    ax1.set_title("Сумарне запізнення", fontsize=12, fontweight="bold")
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # --- Subplot 2: Total Travel Time ---
    for i, algo in enumerate(algorithms):
        values = []
        for scen in scenarios:
            matched = [
                r for r in results
                if r.algorithm == algo and r.scenario == scen
            ]
            values.append(matched[0].total_travel_time if matched else 0)

        bars = ax2.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=LABELS[algo],
            color=COLORS[algo],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax2.set_xlabel("Сценарій / Scenario", fontsize=10)
    ax2.set_ylabel("Час маршруту / Duration (хв)", fontsize=10)
    ax2.set_title("Загальний час маршруту", fontsize=12, fontweight="bold")
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels(scenarios, fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Порівняння метрик якості / Quality Metrics Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = output_dir / "metrics_lateness_duration.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Збережено: %s", path)
    return path


# ---------------------------------------------------------------------------
# 3. CPU Runtime Comparison
# ---------------------------------------------------------------------------


def plot_cpu_runtime(
    results: List[AlgorithmMetrics], output_dir: Path,
) -> Path:
    """Стовпчаста діаграма часу виконання (CPU) для кожного алгоритму."""
    scenarios = sorted({r.scenario for r in results})
    algorithms = ["greedy", "standard_ga", "hybrid_ga"]
    n_scenarios = len(scenarios)

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_scenarios)
    bar_width = 0.22

    for i, algo in enumerate(algorithms):
        values = []
        for scen in scenarios:
            matched = [
                r for r in results
                if r.algorithm == algo and r.scenario == scen
            ]
            values.append(matched[0].execution_time_ms if matched else 0)

        bars = ax.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=LABELS[algo],
            color=COLORS[algo],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Сценарій / Scenario", fontsize=11)
    ax.set_ylabel("Час виконання / CPU Time (мс)", fontsize=11)
    ax.set_title(
        "Час виконання алгоритмів / CPU Runtime Comparison",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    path = output_dir / "cpu_runtime.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Збережено: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Точка входу: запуск бенчмарку та генерація графіків."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TD-VRPTW-P Benchmark — Генерація графіків")
    logger.info("Сценарії: %s", [s.name for s in SCENARIOS])
    logger.info("GA: generations=%d, pop_size=%d, seed=%d",
                GA_GENERATIONS, GA_POP_SIZE, SEED)
    logger.info("Вихідна директорія: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # Запуск бенчмарку
    runner = BenchmarkRunner(
        scenarios=SCENARIOS,
        ga_generations=GA_GENERATIONS,
        ga_pop_size=GA_POP_SIZE,
        seed=SEED,
    )
    results = runner.run_all()

    # Зведена таблиця
    df = runner.to_dataframe()
    logger.info("\nЗведена таблиця результатів:\n%s", df.to_string(index=False))

    # Збереження CSV
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    runner.to_csv(csv_path)

    # Генерація графіків
    logger.info("\nГенерація графіків...")
    plot_convergence(results, OUTPUT_DIR)
    plot_metrics_lateness_duration(results, OUTPUT_DIR)
    plot_cpu_runtime(results, OUTPUT_DIR)

    logger.info("\n✓ Готово! Усі графіки збережено у %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
Модуль генерації часово-залежних матриць тривалості переїздів t_ij(T).

Реалізує:
  1. Haversine-відстань між парами WGS-84 координат.
  2. Динамічний профіль трафіку k(T) ≥ 1.0 на базі суми двох гаусіан
     (ранковий пік 08:00-10:00, вечірній пік 17:00-19:30).
  3. Сервіс ``TrafficMatrixGenerator`` для побудови NxN матриці часу
     переїзду з урахуванням часу виїзду.

Математична основа (ALGORITHMS_SPEC.md):
  - Кожному ребру (i,j) ∈ E відповідає t_ij(T) = d_ij / (v_free / k(T))
    де d_ij — Haversine-відстань, v_free — швидкість вільного руху,
    k(T) — коефіцієнт сповільнення.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from numpy.typing import NDArray

from app.schemas.models import Location

# ---------------------------------------------------------------------------
# Константи
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6371.0
"""Середній радіус Землі (км) для формули гаверсинуса."""

DEFAULT_FREE_FLOW_SPEED_KMH: float = 40.0
"""Базова швидкість вільного руху в місті (км/год)."""

# ---------------------------------------------------------------------------
# Параметри гаусового профілю трафіку
# ---------------------------------------------------------------------------
# Ранковий пік:  μ₁ = 09:00 (540 хв), σ₁ = 30 хв, амплітуда A₁ = 1.0
# Вечірній пік:  μ₂ = 18:15 (1095 хв), σ₂ = 37.5 хв, амплітуда A₂ = 1.2
#
# k(T) = 1.0 + A₁·exp(-(T-μ₁)²/(2σ₁²)) + A₂·exp(-(T-μ₂)²/(2σ₂²))
#
# Це дає:
#   - k(09:00) ≈ 2.0  (ранковий максимум)
#   - k(18:15) ≈ 2.2  (вечірній максимум)
#   - k(03:00) ≈ 1.0  (нічний мінімум)

_MORNING_PEAK_MU: float = 540.0      # 09:00 у хвилинах
_MORNING_PEAK_SIGMA: float = 30.0    # σ₁
_MORNING_PEAK_AMP: float = 1.0       # A₁

_EVENING_PEAK_MU: float = 1095.0     # 18:15 у хвилинах
_EVENING_PEAK_SIGMA: float = 37.5    # σ₂
_EVENING_PEAK_AMP: float = 1.2       # A₂


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def haversine_distance_km(loc_a: Location, loc_b: Location) -> float:
    """Обчислює відстань між двома точками за формулою гаверсинуса.

    Parameters
    ----------
    loc_a, loc_b : Location
        Географічні точки з координатами WGS-84.

    Returns
    -------
    float
        Відстань у кілометрах (≥ 0).

    Notes
    -----
    Формула гаверсинуса:
        a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
        c = 2·atan2(√a, √(1−a))
        d = R·c
    """
    lat1 = math.radians(loc_a.latitude)
    lat2 = math.radians(loc_b.latitude)
    dlat = math.radians(loc_b.latitude - loc_a.latitude)
    dlon = math.radians(loc_b.longitude - loc_a.longitude)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Динамічний профіль трафіку k(T)
# ---------------------------------------------------------------------------

def _gaussian(t: float, mu: float, sigma: float) -> float:
    """Одна гаусова функція exp(-(t-μ)²/(2σ²))."""
    return math.exp(-((t - mu) ** 2) / (2.0 * sigma ** 2))


def traffic_coefficient(departure_minutes: float) -> float:
    """Повертає коефіцієнт сповільнення k(T) ≥ 1.0.

    Parameters
    ----------
    departure_minutes : float
        Час доби у хвилинах від півночі, діапазон [0, 1440).
        Приклади: 0.0 = 00:00, 510.0 = 08:30, 1080.0 = 18:00.

    Returns
    -------
    float
        Коефіцієнт k(T) ≥ 1.0.  Приблизні значення:
        - k(03:00) ≈ 1.00  (нічний мінімум)
        - k(08:30) ≈ 1.97  (ранковий пік)
        - k(09:00) ≈ 2.00  (ранковий максимум)
        - k(18:15) ≈ 2.20  (вечірній максимум)
        - k(12:00) ≈ 1.00  (міжпіковий період)
    """
    # Нормалізуємо до [0, 1440)
    t = departure_minutes % 1440.0

    morning = _MORNING_PEAK_AMP * _gaussian(t, _MORNING_PEAK_MU, _MORNING_PEAK_SIGMA)
    evening = _EVENING_PEAK_AMP * _gaussian(t, _EVENING_PEAK_MU, _EVENING_PEAK_SIGMA)

    return 1.0 + morning + evening


# ---------------------------------------------------------------------------
# TrafficMatrixGenerator
# ---------------------------------------------------------------------------

class TrafficMatrixGenerator:
    """Генератор часово-залежних матриць тривалості переїздів.

    Для пари точок (i, j) та часу виїзду T обчислюється:

        t_ij(T) = (d_ij / v_free) · k(T) · 60   [хвилини]

    де d_ij — Haversine-відстань (км), v_free — вільна швидкість (км/год),
    k(T) — динамічний коефіцієнт трафіку.

    Parameters
    ----------
    free_flow_speed_kmh : float
        Базова швидкість вільного руху (км/год).  За замовчуванням 40.
    """

    def __init__(
        self,
        free_flow_speed_kmh: float = DEFAULT_FREE_FLOW_SPEED_KMH,
    ) -> None:
        if free_flow_speed_kmh <= 0:
            raise ValueError(
                f"free_flow_speed_kmh має бути > 0, отримано {free_flow_speed_kmh}"
            )
        self._speed: float = free_flow_speed_kmh

    # -- публічні методи ---------------------------------------------------

    def get_travel_time(
        self,
        from_loc: Location,
        to_loc: Location,
        departure_time: float,
    ) -> float:
        """Повертає час переїзду між двома точками (у хвилинах).

        Parameters
        ----------
        from_loc : Location
            Точка відправлення.
        to_loc : Location
            Точка призначення.
        departure_time : float
            Час виїзду у хвилинах від початку доби [0, 1440).

        Returns
        -------
        float
            Час переїзду у хвилинах (≥ 0).
            Повертає 0.0, якщо from_loc та to_loc збігаються за координатами.
        """
        dist_km = haversine_distance_km(from_loc, to_loc)
        if dist_km == 0.0:
            return 0.0

        k = traffic_coefficient(departure_time)
        # t = (d / v) * 60 хв/год * k
        travel_hours = dist_km / self._speed
        return travel_hours * 60.0 * k

    def build_matrix(
        self,
        locations: List[Location],
        departure_time: float,
    ) -> NDArray[np.float64]:
        """Будує NxN матрицю часу переїзду для заданих локацій.

        Parameters
        ----------
        locations : List[Location]
            Впорядкований список локацій (v₀, v₁, …, vₙ).
        departure_time : float
            Час виїзду у хвилинах від початку доби [0, 1440).

        Returns
        -------
        NDArray[np.float64]
            Матриця розмірності (N, N), де елемент [i][j] — час переїзду
            від locations[i] до locations[j] у хвилинах.
            Діагональні елементи дорівнюють 0.0.

        Raises
        ------
        ValueError
            Якщо список локацій порожній.

        Notes
        -----
        Матриця побудована за допомогою векторизованого Haversine на numpy
        для ефективної обробки великих масивів локацій.
        """
        n = len(locations)
        if n == 0:
            raise ValueError("Список локацій не може бути порожнім")

        # Витягуємо координати у numpy-масиви
        lats = np.array([loc.latitude for loc in locations], dtype=np.float64)
        lons = np.array([loc.longitude for loc in locations], dtype=np.float64)

        # Переводимо у радіани
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)

        # Різниці координат (N x N)
        dlat = lats_rad[:, np.newaxis] - lats_rad[np.newaxis, :]
        dlon = lons_rad[:, np.newaxis] - lons_rad[np.newaxis, :]

        # Векторизований Haversine
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lats_rad[:, np.newaxis])
            * np.cos(lats_rad[np.newaxis, :])
            * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist_km_matrix = EARTH_RADIUS_KM * c

        # Час переїзду: t = (d / v) * 60 * k(T)
        k = traffic_coefficient(departure_time)
        travel_time_matrix = (dist_km_matrix / self._speed) * 60.0 * k

        # Гарантуємо нулі на діагоналі (усунення числових артефактів)
        np.fill_diagonal(travel_time_matrix, 0.0)

        return travel_time_matrix

"""
Тести модуля генерації часово-залежних матриць тривалості переїздів.

Покриває:
  - Формулу Haversine на контрольних точках Києва
  - Нульову відстань при i == j
  - Профіль трафіку k(T): ранковий/вечірній піки, нічний мінімум
  - TrafficMatrixGenerator: get_travel_time, build_matrix
  - Несиметричність за часом: пік > міжпіковий період
  - Коректність розмірності матриці
  - Діагональні нулі
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.traffic import (
    EARTH_RADIUS_KM,
    TrafficMatrixGenerator,
    haversine_distance_km,
    traffic_coefficient,
)
from app.schemas.models import Location


# =====================================================================
# Фікстури: реальні локації у Києві
# =====================================================================


@pytest.fixture
def kpi() -> Location:
    """КПІ ім. Ігоря Сікорського."""
    return Location(id="kpi", latitude=50.4488, longitude=30.4571, name="КПІ")


@pytest.fixture
def khreshchatyk() -> Location:
    """Хрещатик (центр Києва)."""
    return Location(id="khr", latitude=50.4501, longitude=30.5234, name="Хрещатик")


@pytest.fixture
def obolon() -> Location:
    """Оболонь (північний Київ)."""
    return Location(id="obl", latitude=50.5010, longitude=30.4985, name="Оболонь")


@pytest.fixture
def generator() -> TrafficMatrixGenerator:
    """Генератор з дефолтною швидкістю 40 км/год."""
    return TrafficMatrixGenerator()


# =====================================================================
# Haversine — формула гаверсинуса
# =====================================================================


class TestHaversine:
    """Тести коректності обчислення Haversine-відстані."""

    def test_same_point_zero_distance(self, kpi: Location) -> None:
        """Відстань від точки до самої себе = 0."""
        assert haversine_distance_km(kpi, kpi) == 0.0

    def test_kyiv_kpi_to_khreshchatyk(
        self, kpi: Location, khreshchatyk: Location
    ) -> None:
        """КПІ → Хрещатик ≈ 5-6 км (реальна відстань по прямій)."""
        dist = haversine_distance_km(kpi, khreshchatyk)
        assert 4.5 <= dist <= 7.0, f"Очікувано 4.5-7.0 км, отримано {dist:.2f} км"

    def test_kyiv_kpi_to_obolon(self, kpi: Location, obolon: Location) -> None:
        """КПІ → Оболонь ≈ 6-8 км по прямій."""
        dist = haversine_distance_km(kpi, obolon)
        assert 5.0 <= dist <= 9.0, f"Очікувано 5-9 км, отримано {dist:.2f} км"

    def test_symmetry(self, kpi: Location, khreshchatyk: Location) -> None:
        """Haversine-відстань симетрична: d(A,B) == d(B,A)."""
        d_ab = haversine_distance_km(kpi, khreshchatyk)
        d_ba = haversine_distance_km(khreshchatyk, kpi)
        assert d_ab == pytest.approx(d_ba, abs=1e-10)

    def test_known_distance_poles(self) -> None:
        """Відстань між полюсами ≈ π·R ≈ 20015.09 км."""
        north = Location(id="np", latitude=90.0, longitude=0.0)
        south = Location(id="sp", latitude=-90.0, longitude=0.0)
        dist = haversine_distance_km(north, south)
        expected = math.pi * EARTH_RADIUS_KM
        assert dist == pytest.approx(expected, rel=1e-6)

    def test_equator_one_degree_longitude(self) -> None:
        """На екваторі 1° довготи ≈ 111.19 км."""
        a = Location(id="a", latitude=0.0, longitude=0.0)
        b = Location(id="b", latitude=0.0, longitude=1.0)
        dist = haversine_distance_km(a, b)
        assert 110.0 <= dist <= 112.0, f"Очікувано ~111 км, отримано {dist:.2f}"


# =====================================================================
# Профіль трафіку k(T)
# =====================================================================


class TestTrafficCoefficient:
    """Тести динамічного коефіцієнта сповільнення k(T)."""

    def test_always_ge_one(self) -> None:
        """k(T) ≥ 1.0 для будь-якого часу доби."""
        for minute in range(0, 1440, 5):
            k = traffic_coefficient(float(minute))
            assert k >= 1.0, f"k({minute}) = {k} < 1.0"

    def test_night_minimum(self) -> None:
        """Нічний час (03:00) → k ≈ 1.0."""
        k = traffic_coefficient(180.0)  # 03:00
        assert k == pytest.approx(1.0, abs=0.01)

    def test_midday_low(self) -> None:
        """Міжпіковий час (12:00) → k ≈ 1.0."""
        k = traffic_coefficient(720.0)  # 12:00
        assert k == pytest.approx(1.0, abs=0.05)

    def test_morning_peak_range(self) -> None:
        """Ранковий пік (08:30-09:00) → k ∈ [1.6, 2.0]."""
        k_0830 = traffic_coefficient(510.0)  # 08:30
        k_0900 = traffic_coefficient(540.0)  # 09:00
        assert 1.6 <= k_0830 <= 2.1, f"k(08:30) = {k_0830}"
        assert 1.9 <= k_0900 <= 2.1, f"k(09:00) = {k_0900}"

    def test_evening_peak_range(self) -> None:
        """Вечірній пік (18:00-18:30) → k ∈ [1.7, 2.2]."""
        k_1800 = traffic_coefficient(1080.0)  # 18:00
        k_1815 = traffic_coefficient(1095.0)  # 18:15
        assert 1.7 <= k_1800 <= 2.3, f"k(18:00) = {k_1800}"
        assert 2.0 <= k_1815 <= 2.3, f"k(18:15) = {k_1815}"

    def test_morning_higher_than_night(self) -> None:
        """Ранковий пік суттєво вищий за нічний час."""
        k_peak = traffic_coefficient(540.0)   # 09:00
        k_night = traffic_coefficient(180.0)  # 03:00
        assert k_peak > k_night + 0.5

    def test_wraps_around_1440(self) -> None:
        """Час >= 1440 хв нормалізується (модульна арифметика)."""
        k_normal = traffic_coefficient(540.0)
        k_wrapped = traffic_coefficient(540.0 + 1440.0)  # +24 год
        assert k_normal == pytest.approx(k_wrapped, abs=1e-10)


# =====================================================================
# TrafficMatrixGenerator — get_travel_time
# =====================================================================


class TestGetTravelTime:
    """Тести обчислення часу переїзду між парою точок."""

    def test_same_location_zero(
        self, generator: TrafficMatrixGenerator, kpi: Location
    ) -> None:
        """Час переїзду від точки до самої себе = 0."""
        tt = generator.get_travel_time(kpi, kpi, departure_time=510.0)
        assert tt == 0.0

    def test_positive_for_different_points(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
    ) -> None:
        """Час переїзду між різними точками > 0."""
        tt = generator.get_travel_time(kpi, khreshchatyk, departure_time=720.0)
        assert tt > 0.0

    def test_peak_slower_than_night(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
    ) -> None:
        """Переїзд о 08:30 (пік) повільніший, ніж о 03:00 (ніч)."""
        tt_peak = generator.get_travel_time(kpi, khreshchatyk, departure_time=510.0)
        tt_night = generator.get_travel_time(kpi, khreshchatyk, departure_time=180.0)
        assert tt_peak > tt_night, (
            f"Пік ({tt_peak:.2f} хв) має бути > ніч ({tt_night:.2f} хв)"
        )

    def test_evening_peak_slower_than_midday(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        obolon: Location,
    ) -> None:
        """Переїзд о 18:15 (вечірній пік) повільніший, ніж о 12:00."""
        tt_evening = generator.get_travel_time(kpi, obolon, departure_time=1095.0)
        tt_midday = generator.get_travel_time(kpi, obolon, departure_time=720.0)
        assert tt_evening > tt_midday

    def test_reasonable_travel_time_kyiv(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
    ) -> None:
        """КПІ → Хрещатик (~5 км) при вільному русі (40 км/год) ≈ 7-8 хв."""
        tt = generator.get_travel_time(kpi, khreshchatyk, departure_time=180.0)
        # 5 км / 40 км/год = 0.125 год = 7.5 хв (± допуск)
        assert 5.0 <= tt <= 12.0, f"Очікувано 5-12 хв, отримано {tt:.2f} хв"

    def test_custom_speed(self, kpi: Location, khreshchatyk: Location) -> None:
        """Зменшення швидкості збільшує час переїзду."""
        gen_slow = TrafficMatrixGenerator(free_flow_speed_kmh=20.0)
        gen_fast = TrafficMatrixGenerator(free_flow_speed_kmh=60.0)
        tt_slow = gen_slow.get_travel_time(kpi, khreshchatyk, departure_time=720.0)
        tt_fast = gen_fast.get_travel_time(kpi, khreshchatyk, departure_time=720.0)
        assert tt_slow > tt_fast

    def test_invalid_speed_raises(self) -> None:
        """Від'ємна або нульова швидкість — ValueError."""
        with pytest.raises(ValueError, match="має бути > 0"):
            TrafficMatrixGenerator(free_flow_speed_kmh=0.0)
        with pytest.raises(ValueError, match="має бути > 0"):
            TrafficMatrixGenerator(free_flow_speed_kmh=-10.0)


# =====================================================================
# TrafficMatrixGenerator — build_matrix
# =====================================================================


class TestBuildMatrix:
    """Тести побудови NxN матриці часу переїзду."""

    def test_matrix_shape(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
        obolon: Location,
    ) -> None:
        """Матриця для N=3 локацій має форму (3, 3)."""
        locs = [kpi, khreshchatyk, obolon]
        matrix = generator.build_matrix(locs, departure_time=720.0)
        assert matrix.shape == (3, 3)

    def test_matrix_dtype(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
    ) -> None:
        """Матриця має тип float64."""
        matrix = generator.build_matrix([kpi, khreshchatyk], departure_time=0.0)
        assert matrix.dtype == np.float64

    def test_diagonal_zeros(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
        obolon: Location,
    ) -> None:
        """Діагональні елементи (i==j) дорівнюють 0."""
        locs = [kpi, khreshchatyk, obolon]
        matrix = generator.build_matrix(locs, departure_time=510.0)
        for i in range(len(locs)):
            assert matrix[i, i] == 0.0, f"Діагональ [{i},{i}] != 0"

    def test_off_diagonal_positive(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
        obolon: Location,
    ) -> None:
        """Позадіагональні елементи > 0 для різних точок."""
        locs = [kpi, khreshchatyk, obolon]
        matrix = generator.build_matrix(locs, departure_time=720.0)
        for i in range(len(locs)):
            for j in range(len(locs)):
                if i != j:
                    assert matrix[i, j] > 0.0, f"[{i},{j}] має бути > 0"

    def test_symmetric_distances(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
        obolon: Location,
    ) -> None:
        """При фіксованому часі виїзду матриця симетрична (бо Haversine симетрична)."""
        locs = [kpi, khreshchatyk, obolon]
        matrix = generator.build_matrix(locs, departure_time=720.0)
        np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=10)

    def test_single_location_matrix(
        self, generator: TrafficMatrixGenerator, kpi: Location
    ) -> None:
        """Матриця для однієї локації — [[0.0]]."""
        matrix = generator.build_matrix([kpi], departure_time=0.0)
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0.0

    def test_peak_matrix_values_larger(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
        obolon: Location,
    ) -> None:
        """Матриця в пік (08:30) > матриця вночі (03:00) поелементно."""
        locs = [kpi, khreshchatyk, obolon]
        m_peak = generator.build_matrix(locs, departure_time=510.0)
        m_night = generator.build_matrix(locs, departure_time=180.0)

        for i in range(len(locs)):
            for j in range(len(locs)):
                if i != j:
                    assert m_peak[i, j] > m_night[i, j], (
                        f"Пік [{i},{j}]={m_peak[i,j]:.2f} має бути > "
                        f"ніч [{i},{j}]={m_night[i,j]:.2f}"
                    )

    def test_consistent_with_get_travel_time(
        self,
        generator: TrafficMatrixGenerator,
        kpi: Location,
        khreshchatyk: Location,
        obolon: Location,
    ) -> None:
        """Елементи матриці мають відповідати get_travel_time."""
        locs = [kpi, khreshchatyk, obolon]
        dep = 600.0
        matrix = generator.build_matrix(locs, departure_time=dep)

        for i, loc_i in enumerate(locs):
            for j, loc_j in enumerate(locs):
                expected = generator.get_travel_time(loc_i, loc_j, dep)
                assert matrix[i, j] == pytest.approx(expected, rel=1e-9), (
                    f"Матриця [{i},{j}]={matrix[i,j]:.6f} ≠ "
                    f"get_travel_time={expected:.6f}"
                )

    def test_empty_locations_raises(
        self, generator: TrafficMatrixGenerator
    ) -> None:
        """Порожній список локацій — ValueError."""
        with pytest.raises(ValueError, match="порожнім"):
            generator.build_matrix([], departure_time=0.0)

    def test_large_matrix_shape(self, generator: TrafficMatrixGenerator) -> None:
        """Матриця для N=20 локацій має коректну форму."""
        locs = [
            Location(id=f"loc-{i}", latitude=50.0 + i * 0.01, longitude=30.0 + i * 0.01)
            for i in range(20)
        ]
        matrix = generator.build_matrix(locs, departure_time=720.0)
        assert matrix.shape == (20, 20)
        assert np.all(np.diag(matrix) == 0.0)

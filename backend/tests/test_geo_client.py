"""
Тести асинхронного геоклієнта маршрутизації.

Покриває:
  - Парсинг валідної відповіді OSRM Table API (мок HTTP)
  - Graceful Fallback при помилці 500 від OSRM
  - Graceful Fallback при таймауті OSRM
  - TTL-кешування (повторний виклик не робить новий HTTP-запит)
  - Валідація: відмова при < 2 локаціях
  - SyntheticClient: коректність матриць Haversine
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest
import respx

from app.core.config import GeoSettings
from app.core.geo_client import (
    OSRMClient,
    OSRMResponseError,
    RoutingServiceManager,
    SyntheticClient,
    create_routing_manager,
)
from app.schemas.models import Location


# =====================================================================
# Фікстури
# =====================================================================


@pytest.fixture
def kyiv_locations() -> list[Location]:
    """Три локації у Києві для тестування."""
    return [
        Location(id="kpi", latitude=50.4488, longitude=30.4571, name="КПІ"),
        Location(
            id="maidan",
            latitude=50.4501,
            longitude=30.5234,
            name="Майдан Незалежності",
        ),
        Location(
            id="obolon",
            latitude=50.5010,
            longitude=30.4980,
            name="Оболонь",
        ),
    ]


@pytest.fixture
def osrm_valid_response_3x3() -> dict:
    """Валідна відповідь OSRM Table API для 3 точок."""
    return {
        "code": "Ok",
        "durations": [
            [0.0, 600.5, 900.2],
            [610.3, 0.0, 450.8],
            [880.1, 430.6, 0.0],
        ],
        "distances": [
            [0.0, 5200.0, 8100.0],
            [5300.0, 0.0, 3900.0],
            [8000.0, 3800.0, 0.0],
        ],
        "sources": [],
        "destinations": [],
    }


@pytest.fixture
def osrm_client() -> OSRMClient:
    """OSRMClient з мінімальними retry для швидких тестів."""
    return OSRMClient(
        base_url="http://test-osrm.local",
        timeout=5.0,
        max_retries=1,
    )


@pytest.fixture
def synthetic_client() -> SyntheticClient:
    """SyntheticClient з фіксованим часом виїзду."""
    return SyntheticClient(departure_time=540.0)


# =====================================================================
# Тест 1: Успішний парсинг відповіді OSRM
# =====================================================================


@pytest.mark.asyncio
@respx.mock
async def test_osrm_success_parsing(
    kyiv_locations: list[Location],
    osrm_valid_response_3x3: dict,
    osrm_client: OSRMClient,
) -> None:
    """Перевіряє коректний парсинг валідної відповіді OSRM Table API."""
    # Мокаємо OSRM Table endpoint
    expected_coords = ";".join(
        f"{loc.longitude},{loc.latitude}" for loc in kyiv_locations
    )
    route = respx.get(
        f"http://test-osrm.local/table/v1/driving/{expected_coords}",
        params={"annotations": "duration,distance"},
    ).respond(json=osrm_valid_response_3x3)

    durations, distances = await osrm_client.get_distance_matrix(
        kyiv_locations,
    )

    # Запит мав відбутися рівно 1 раз
    assert route.called
    assert route.call_count == 1

    # Перевіряємо розмірність
    assert durations.shape == (3, 3)
    assert distances.shape == (3, 3)

    # Перевіряємо конкретні значення
    assert durations[0, 0] == 0.0
    assert durations[0, 1] == pytest.approx(600.5)
    assert distances[0, 1] == pytest.approx(5200.0)

    # Діагональ — нулі
    for i in range(3):
        assert durations[i, i] == 0.0
        assert distances[i, i] == 0.0


# =====================================================================
# Тест 2: Fallback при помилці 500
# =====================================================================


@pytest.mark.asyncio
@respx.mock
async def test_fallback_on_500_error(
    kyiv_locations: list[Location],
) -> None:
    """При HTTP 500 від OSRM менеджер повертає синтетичну матрицю."""
    # Мокаємо OSRM — повертає 500
    respx.get(
        url__startswith="http://test-osrm.local/table/v1/driving/",
    ).respond(status_code=500, text="Internal Server Error")

    primary = OSRMClient(
        base_url="http://test-osrm.local",
        timeout=5.0,
        max_retries=1,
    )
    fallback = SyntheticClient(departure_time=540.0)
    manager = RoutingServiceManager(
        primary=primary, fallback=fallback, cache_ttl=3600,
    )

    # Не повинно впасти — має перемкнутися на fallback
    durations, distances = await manager.get_distance_matrix(kyiv_locations)

    # Результат — валідні numpy-масиви від SyntheticClient
    assert isinstance(durations, np.ndarray)
    assert isinstance(distances, np.ndarray)
    assert durations.shape == (3, 3)
    assert distances.shape == (3, 3)

    # Діагональ — нулі
    for i in range(3):
        assert durations[i, i] == 0.0
        assert distances[i, i] == 0.0

    # Позадіагональні елементи > 0
    assert durations[0, 1] > 0.0
    assert distances[0, 1] > 0.0


# =====================================================================
# Тест 3: Fallback при таймауті
# =====================================================================


@pytest.mark.asyncio
@respx.mock
async def test_fallback_on_timeout(
    kyiv_locations: list[Location],
) -> None:
    """При таймауті OSRM менеджер повертає синтетичну матрицю."""
    # Мокаємо OSRM — кидає ConnectError (імітація таймауту)
    respx.get(
        url__startswith="http://test-osrm.local/table/v1/driving/",
    ).mock(side_effect=httpx.ConnectTimeout("Connection timed out"))

    primary = OSRMClient(
        base_url="http://test-osrm.local",
        timeout=1.0,
        max_retries=1,
    )
    fallback = SyntheticClient(departure_time=540.0)
    manager = RoutingServiceManager(
        primary=primary, fallback=fallback, cache_ttl=3600,
    )

    durations, distances = await manager.get_distance_matrix(kyiv_locations)

    assert durations.shape == (3, 3)
    assert distances.shape == (3, 3)
    assert durations[0, 1] > 0.0


# =====================================================================
# Тест 4: Кешування — повторний виклик не робить HTTP-запит
# =====================================================================


@pytest.mark.asyncio
@respx.mock
async def test_cache_hit_no_second_request(
    kyiv_locations: list[Location],
    osrm_valid_response_3x3: dict,
) -> None:
    """Другий виклик з тими самими локаціями бере результат із кешу."""
    route = respx.get(
        url__startswith="http://test-osrm.local/table/v1/driving/",
    ).respond(json=osrm_valid_response_3x3)

    primary = OSRMClient(
        base_url="http://test-osrm.local",
        timeout=5.0,
        max_retries=1,
    )
    fallback = SyntheticClient(departure_time=540.0)
    manager = RoutingServiceManager(
        primary=primary, fallback=fallback, cache_ttl=3600,
    )

    # Перший виклик — HTTP-запит
    result1 = await manager.get_distance_matrix(kyiv_locations)
    assert route.call_count == 1

    # Другий виклик — з кешу (HTTP-запиту бути не повинно)
    result2 = await manager.get_distance_matrix(kyiv_locations)
    assert route.call_count == 1  # Не збільшився

    # Результати ідентичні
    np.testing.assert_array_equal(result1[0], result2[0])
    np.testing.assert_array_equal(result1[1], result2[1])


# =====================================================================
# Тест 5: Валідація — менше 2 локацій
# =====================================================================


@pytest.mark.asyncio
async def test_validation_too_few_locations() -> None:
    """ValueError при спробі побудувати матрицю з < 2 точок."""
    single = [Location(id="a", latitude=50.0, longitude=30.0)]
    empty: list[Location] = []

    primary = OSRMClient(base_url="http://test-osrm.local")
    fallback = SyntheticClient()
    manager = RoutingServiceManager(
        primary=primary, fallback=fallback,
    )

    with pytest.raises(ValueError, match="щонайменше 2"):
        await manager.get_distance_matrix(single)

    with pytest.raises(ValueError, match="щонайменше 2"):
        await manager.get_distance_matrix(empty)


# =====================================================================
# Тест 6: SyntheticClient — коректність матриць
# =====================================================================


@pytest.mark.asyncio
async def test_synthetic_client_directly(
    kyiv_locations: list[Location],
    synthetic_client: SyntheticClient,
) -> None:
    """SyntheticClient повертає валідні матриці Haversine."""
    durations, distances = await synthetic_client.get_distance_matrix(
        kyiv_locations,
    )

    n = len(kyiv_locations)

    # Розмірність
    assert durations.shape == (n, n)
    assert distances.shape == (n, n)

    # Діагональ — нулі
    for i in range(n):
        assert durations[i, i] == 0.0
        assert distances[i, i] == 0.0

    # Позадіагональні елементи > 0
    for i in range(n):
        for j in range(n):
            if i != j:
                assert durations[i, j] > 0.0, (
                    f"durations[{i},{j}] повинно бути > 0"
                )
                assert distances[i, j] > 0.0, (
                    f"distances[{i},{j}] повинно бути > 0"
                )

    # Тривалості — у секундах (> 60, бо відстань КПІ-Майдан ≈ 5 км)
    assert durations[0, 1] > 60.0

    # Відстані — у метрах (КПІ-Майдан ≈ 5 км → > 4000 м)
    assert distances[0, 1] > 4000.0

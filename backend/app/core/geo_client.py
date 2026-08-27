"""
Асинхронний клієнт маршрутизації з кешуванням та Graceful Fallback.

Модуль забезпечує єдиний інтерфейс для отримання матриць тривалості
та відстаней між локаціями.  Підтримує OSRM Table API як основне
джерело та синтетичний Haversine-розрахунок як резервне.

Архітектура
-----------
- ``BaseRoutingClient`` — протокол (інтерфейс) для клієнтів маршрутизації.
- ``OSRMClient`` — асинхронний клієнт до OSRM Table Service.
- ``SyntheticClient`` — обгортка над ``TrafficMatrixGenerator`` (Haversine).
- ``RoutingServiceManager`` — фасад із TTL-кешуванням та автоматичним
  перемиканням на ``SyntheticClient`` при помилках зовнішнього API.

Використання
------------
::

    from app.core.geo_client import create_routing_manager

    manager = create_routing_manager()
    durations, distances = await manager.get_distance_matrix(locations)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx
import numpy as np
from numpy.typing import NDArray

from app.core.config import GeoSettings
from app.core.traffic import (
    EARTH_RADIUS_KM,
    TrafficMatrixGenerator,
    haversine_distance_km,
)
from app.schemas.models import Location

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Типи
# ---------------------------------------------------------------------------

DistanceMatrix = NDArray[np.float64]
"""NxN матриця відстаней (метри)."""

DurationMatrix = NDArray[np.float64]
"""NxN матриця тривалостей (секунди)."""


# ---------------------------------------------------------------------------
# Виняткові ситуації
# ---------------------------------------------------------------------------


class GeoClientError(Exception):
    """Базовий виняток геоклієнта."""


class OSRMRequestError(GeoClientError):
    """Помилка HTTP-запиту до OSRM."""


class OSRMResponseError(GeoClientError):
    """Невалідна або помилкова відповідь від OSRM."""


# ---------------------------------------------------------------------------
# Протокол (інтерфейс) клієнта маршрутизації
# ---------------------------------------------------------------------------


class BaseRoutingClient(Protocol):
    """Протокол асинхронного клієнта маршрутизації.

    Кожна реалізація повинна повертати кортеж
    ``(duration_matrix, distance_matrix)``:
    - ``duration_matrix`` — NxN ``np.ndarray``, значення у **секундах**.
    - ``distance_matrix`` — NxN ``np.ndarray``, значення у **метрах**.
    """

    async def get_distance_matrix(
        self, locations: List[Location],
    ) -> Tuple[DurationMatrix, DistanceMatrix]:
        """Повертає матрицю тривалостей та відстаней між локаціями."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# OSRMClient
# ---------------------------------------------------------------------------


class OSRMClient:
    """Асинхронний клієнт до OSRM Table Service.

    Формує запит до ``/table/v1/driving/{coordinates}``
    з анотаціями ``duration`` та ``distance``, парсить відповідь
    у пару numpy-масивів.

    Parameters
    ----------
    base_url : str
        Базовий URL OSRM (без ``/table/...``).
    timeout : float
        Таймаут одного HTTP-запиту (секунди).
    max_retries : int
        Кількість повторних спроб з exponential backoff.
    """

    _BACKOFF_BASE: float = 0.5
    """Початкова затримка backoff (секунди)."""

    _BACKOFF_FACTOR: float = 2.0
    """Множник для exponential backoff."""

    def __init__(
        self,
        base_url: str = "http://router.project-osrm.org",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

    # -- публічний інтерфейс ------------------------------------------------

    async def get_distance_matrix(
        self, locations: List[Location],
    ) -> Tuple[DurationMatrix, DistanceMatrix]:
        """Надсилає запит до OSRM Table API та повертає матриці.

        Parameters
        ----------
        locations : List[Location]
            Список локацій (≥ 2).

        Returns
        -------
        Tuple[DurationMatrix, DistanceMatrix]
            ``(durations_seconds, distances_meters)``

        Raises
        ------
        ValueError
            Якщо ``len(locations) < 2``.
        OSRMRequestError
            Якщо після всіх повторних спроб запит не вдався.
        OSRMResponseError
            Якщо OSRM повернув невалідну відповідь.
        """
        _validate_locations(locations)

        coords_str = self._build_coordinates_string(locations)
        url = (
            f"{self._base_url}/table/v1/driving/{coords_str}"
            f"?annotations=duration,distance"
        )

        data = await self._request_with_retry(url)
        return self._parse_response(data, len(locations))

    # -- внутрішні методи ---------------------------------------------------

    @staticmethod
    def _build_coordinates_string(locations: List[Location]) -> str:
        """Будує рядок координат у форматі OSRM: ``lng,lat;lng,lat;...``

        OSRM очікує координати у порядку **longitude, latitude**.
        """
        parts = [f"{loc.longitude},{loc.latitude}" for loc in locations]
        return ";".join(parts)

    async def _request_with_retry(self, url: str) -> Dict[str, Any]:
        """Виконує HTTP GET із retry та exponential backoff.

        Parameters
        ----------
        url : str
            Повний URL запиту.

        Returns
        -------
        Dict[str, Any]
            Розпарсений JSON-об'єкт відповіді.

        Raises
        ------
        OSRMRequestError
            Якщо усі спроби вичерпані.
        """
        last_exc: Optional[Exception] = None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._max_retries + 1):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.json()

                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        delay = self._BACKOFF_BASE * (
                            self._BACKOFF_FACTOR ** (attempt - 1)
                        )
                        logger.warning(
                            "OSRM запит (спроба %d/%d) не вдався: %s. "
                            "Повтор через %.1f с",
                            attempt,
                            self._max_retries,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "OSRM запит остаточно не вдався після %d спроб: %s",
                            self._max_retries,
                            exc,
                        )

        raise OSRMRequestError(
            f"OSRM запит не вдався після {self._max_retries} спроб: "
            f"{last_exc}"
        ) from last_exc

    @staticmethod
    def _parse_response(
        data: Dict[str, Any], n: int,
    ) -> Tuple[DurationMatrix, DistanceMatrix]:
        """Парсить JSON-відповідь OSRM Table API.

        Parameters
        ----------
        data : Dict[str, Any]
            JSON-об'єкт відповіді OSRM.
        n : int
            Очікувана розмірність матриці (N x N).

        Returns
        -------
        Tuple[DurationMatrix, DistanceMatrix]
            Матриці тривалості (секунди) та відстані (метри).
            Значення ``null`` у відповіді замінюються на ``np.inf``.

        Raises
        ------
        OSRMResponseError
            Якщо ``code != "Ok"`` або матриці відсутні / невалідні.
        """
        code = data.get("code")
        if code != "Ok":
            message = data.get("message", "невідома помилка")
            raise OSRMResponseError(
                f"OSRM повернув code='{code}': {message}"
            )

        raw_durations = data.get("durations")
        raw_distances = data.get("distances")

        if raw_durations is None or raw_distances is None:
            raise OSRMResponseError(
                "Відповідь OSRM не містить матриць 'durations' та/або "
                "'distances'. Переконайтесь, що запит містить "
                "?annotations=duration,distance"
            )

        # Конвертуємо у numpy, замінюючи null → np.inf
        durations = np.array(
            [
                [v if v is not None else np.inf for v in row]
                for row in raw_durations
            ],
            dtype=np.float64,
        )
        distances = np.array(
            [
                [v if v is not None else np.inf for v in row]
                for row in raw_distances
            ],
            dtype=np.float64,
        )

        if durations.shape != (n, n) or distances.shape != (n, n):
            raise OSRMResponseError(
                f"Очікувані матриці розміром ({n}, {n}), отримано "
                f"durations={durations.shape}, distances={distances.shape}"
            )

        return durations, distances


# ---------------------------------------------------------------------------
# SyntheticClient (Haversine fallback)
# ---------------------------------------------------------------------------


class SyntheticClient:
    """Синтетичний клієнт маршрутизації на базі Haversine.

    Використовує ``TrafficMatrixGenerator`` для обчислення матриці
    тривалостей і окремо будує матрицю відстаней (Haversine).

    Parameters
    ----------
    departure_time : float
        Час виїзду у хвилинах від початку доби [0, 1440).
        Використовується для обчислення трафік-коефіцієнту k(T).
    free_flow_speed_kmh : float
        Базова швидкість вільного руху (км/год).
    """

    def __init__(
        self,
        departure_time: float = 540.0,
        free_flow_speed_kmh: float = 40.0,
    ) -> None:
        self._departure_time = departure_time
        self._generator = TrafficMatrixGenerator(
            free_flow_speed_kmh=free_flow_speed_kmh,
        )

    async def get_distance_matrix(
        self, locations: List[Location],
    ) -> Tuple[DurationMatrix, DistanceMatrix]:
        """Повертає синтетичні матриці на базі Haversine.

        Returns
        -------
        Tuple[DurationMatrix, DistanceMatrix]
            - duration_matrix: секунди (TrafficMatrixGenerator → хв × 60)
            - distance_matrix: метри (Haversine × 1000)
        """
        _validate_locations(locations)

        # Матриця тривалостей (хвилини → секунди)
        time_matrix_min = self._generator.build_matrix(
            locations, self._departure_time,
        )
        duration_matrix = time_matrix_min * 60.0

        # Матриця відстаней (км → метри) — векторизований Haversine
        distance_matrix = self._build_distance_matrix_meters(locations)

        return duration_matrix, distance_matrix

    @staticmethod
    def _build_distance_matrix_meters(
        locations: List[Location],
    ) -> DistanceMatrix:
        """Будує NxN матрицю відстаней у метрах (Haversine)."""
        n = len(locations)
        lats = np.array(
            [loc.latitude for loc in locations], dtype=np.float64,
        )
        lons = np.array(
            [loc.longitude for loc in locations], dtype=np.float64,
        )

        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)

        dlat = lats_rad[:, np.newaxis] - lats_rad[np.newaxis, :]
        dlon = lons_rad[:, np.newaxis] - lons_rad[np.newaxis, :]

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lats_rad[:, np.newaxis])
            * np.cos(lats_rad[np.newaxis, :])
            * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist_km = EARTH_RADIUS_KM * c

        dist_meters = dist_km * 1000.0
        np.fill_diagonal(dist_meters, 0.0)

        return dist_meters


# ---------------------------------------------------------------------------
# TTL-кеш (внутрішній)
# ---------------------------------------------------------------------------


class _CacheEntry:
    """Запис у TTL-кеші з часовою міткою створення."""

    __slots__ = ("value", "created_at")

    def __init__(
        self,
        value: Tuple[DurationMatrix, DistanceMatrix],
    ) -> None:
        self.value = value
        self.created_at: float = time.monotonic()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Перевіряє, чи минув TTL."""
        return (time.monotonic() - self.created_at) > ttl_seconds


# ---------------------------------------------------------------------------
# RoutingServiceManager (фасад)
# ---------------------------------------------------------------------------


class RoutingServiceManager:
    """Менеджер маршрутизації з кешуванням та Graceful Fallback.

    Координує первинний клієнт (``OSRMClient``) та резервний
    (``SyntheticClient``).  При будь-якій помилці первинного клієнта
    автоматично перемикається на резервний із логуванням попередження.

    Parameters
    ----------
    primary : BaseRoutingClient
        Основний клієнт маршрутизації (зазвичай ``OSRMClient``).
    fallback : SyntheticClient
        Резервний клієнт (Haversine).
    cache_ttl : int
        Час життя кешу (секунди).
    """

    def __init__(
        self,
        primary: BaseRoutingClient,
        fallback: SyntheticClient,
        cache_ttl: int = 3600,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, _CacheEntry] = {}

    async def get_distance_matrix(
        self, locations: List[Location],
    ) -> Tuple[DurationMatrix, DistanceMatrix]:
        """Повертає матриці тривалостей та відстаней із кешуванням і fallback.

        Логіка:
        1. Перевіряє кеш — якщо є валідний запис, повертає його.
        2. Запитує primary-клієнт (наприклад, OSRM).
        3. При помилці primary → логує warning → викликає fallback.
        4. Результат кешується.

        Parameters
        ----------
        locations : List[Location]
            Список локацій (≥ 2).

        Returns
        -------
        Tuple[DurationMatrix, DistanceMatrix]
            ``(durations_seconds, distances_meters)``

        Raises
        ------
        ValueError
            Якщо ``len(locations) < 2``.
        """
        _validate_locations(locations)

        cache_key = self._make_cache_key(locations)

        # 1. Перевірка кешу
        cached = self._cache.get(cache_key)
        if cached is not None and not cached.is_expired(self._cache_ttl):
            logger.debug("Кеш-хіт для %d локацій", len(locations))
            return cached.value

        # 2. Спроба primary-клієнта
        try:
            result = await self._primary.get_distance_matrix(locations)
            self._cache[cache_key] = _CacheEntry(result)
            logger.info(
                "Матриці отримано від primary-клієнта (%d×%d)",
                len(locations),
                len(locations),
            )
            return result

        except Exception as exc:
            logger.warning(
                "Primary-клієнт маршрутизації не доступний: %s. "
                "Перемикання на синтетичний Haversine fallback.",
                exc,
            )

        # 3. Fallback
        result = await self._fallback.get_distance_matrix(locations)
        self._cache[cache_key] = _CacheEntry(result)
        logger.info(
            "Матриці отримано від fallback-клієнта (Haversine, %d×%d)",
            len(locations),
            len(locations),
        )
        return result

    @staticmethod
    def _make_cache_key(locations: List[Location]) -> str:
        """Генерує детермінований ключ кешу для набору локацій.

        Ключ залежить від порядку та координат точок.
        """
        coords = "|".join(
            f"{loc.latitude:.8f},{loc.longitude:.8f}" for loc in locations
        )
        return hashlib.sha256(coords.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Повністю очищає кеш."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Валідація
# ---------------------------------------------------------------------------


def _validate_locations(locations: List[Location]) -> None:
    """Перевіряє, що список локацій містить щонайменше 2 точки.

    Raises
    ------
    ValueError
        Якщо ``len(locations) < 2``.
    """
    if len(locations) < 2:
        raise ValueError(
            f"Для побудови матриці потрібно щонайменше 2 локації, "
            f"отримано {len(locations)}"
        )


# ---------------------------------------------------------------------------
# Фабрична функція
# ---------------------------------------------------------------------------


def create_routing_manager(
    settings: Optional[GeoSettings] = None,
    departure_time: float = 540.0,
) -> RoutingServiceManager:
    """Створює ``RoutingServiceManager`` за конфігурацією.

    Parameters
    ----------
    settings : Optional[GeoSettings]
        Конфігурація геосервісу.  Якщо ``None`` — створюється
        з env-змінних / значень за замовчуванням.
    departure_time : float
        Час виїзду (хвилини від початку доби) для SyntheticClient.

    Returns
    -------
    RoutingServiceManager
        Готовий до використання менеджер маршрутизації.
    """
    if settings is None:
        settings = GeoSettings()

    fallback = SyntheticClient(departure_time=departure_time)

    provider = settings.GEO_PROVIDER

    if provider == "osrm":
        primary: BaseRoutingClient = OSRMClient(
            base_url=settings.OSRM_BASE_URL,
            timeout=settings.GEO_REQUEST_TIMEOUT,
            max_retries=settings.GEO_MAX_RETRIES,
        )
    elif provider == "synthetic":
        primary = SyntheticClient(departure_time=departure_time)
    else:
        # ors — placeholder для подальшого розширення
        logger.warning(
            "Провайдер '%s' ще не реалізовано повністю, "
            "використовується OSRM як primary.",
            provider,
        )
        primary = OSRMClient(
            base_url=settings.OSRM_BASE_URL,
            timeout=settings.GEO_REQUEST_TIMEOUT,
            max_retries=settings.GEO_MAX_RETRIES,
        )

    return RoutingServiceManager(
        primary=primary,
        fallback=fallback,
        cache_ttl=settings.GEO_CACHE_TTL_SECONDS,
    )

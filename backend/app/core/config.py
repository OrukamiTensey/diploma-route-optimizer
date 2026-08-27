"""
Конфігурація геосервісних параметрів через pydantic-settings.

Параметри зчитуються зі змінних оточення або файлу ``.env``.
Підтримувані провайдери: ``osrm`` (за замовчуванням), ``ors``, ``synthetic``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class GeoSettings(BaseSettings):
    """Налаштування модуля маршрутизації та геосервісів.

    Attributes
    ----------
    GEO_PROVIDER : str
        Обраний провайдер маршрутизації (``osrm``, ``ors``, ``synthetic``).
    OSRM_BASE_URL : str
        Базовий URL OSRM-інстансу.
    ORS_API_KEY : Optional[str]
        API-ключ OpenRouteService (потрібен лише при ``GEO_PROVIDER='ors'``).
    GEO_CACHE_TTL_SECONDS : int
        Час життя кешованої матриці (секунди).
    GEO_REQUEST_TIMEOUT : float
        Таймаут одного HTTP-запиту до геосервісу (секунди).
    GEO_MAX_RETRIES : int
        Максимальна кількість повторних спроб при помилці мережі.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GEO_PROVIDER: Literal["osrm", "ors", "synthetic"] = "osrm"

    OSRM_BASE_URL: str = "http://router.project-osrm.org"

    ORS_API_KEY: Optional[str] = None

    GEO_CACHE_TTL_SECONDS: int = 3600

    GEO_REQUEST_TIMEOUT: float = 30.0

    GEO_MAX_RETRIES: int = 3

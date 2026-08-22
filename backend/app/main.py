"""
Головний модуль FastAPI-застосунку TD-VRPTW Optimization API.

Ініціалізує FastAPI, підключає CORS Middleware та реєструє
роутери версії v1.

Запуск:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router as v1_router

app = FastAPI(
    title="TD-VRPTW Optimization API",
    description=(
        "REST API для інтелектуальної оптимізації міських маршрутів "
        "з урахуванням часових вікон, пріоритетів та динамічного трафіку. "
        "Генетичний алгоритм з адаптивною мутацією + 2-opt локальний пошук."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS Middleware (дозволяємо всі джерела для dev-середовища)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Реєстрація роутерів
# ---------------------------------------------------------------------------

app.include_router(v1_router, prefix="/api/v1")

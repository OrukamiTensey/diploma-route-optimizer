# ===========================================================================
# Dockerfile для TD-VRPTW Optimization API
#
# Використовує python:3.11-slim для легкого образу.
# Збірка: docker build -t td-vrptw-api .
# Запуск:  docker run -p 8000:8000 td-vrptw-api
# ===========================================================================

FROM python:3.11-slim AS base

# Мітки (OCI metadata)
LABEL maintainer="TD-VRPTW Team"
LABEL description="TD-VRPTW Optimization API — FastAPI + GA + 2-opt"

# Системні залежності для NumPy / SciPy / Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        libfreetype6-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Робоча директорія
WORKDIR /app

# Встановлення Python-залежностей (кешований шар)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копіювання коду бекенду
COPY backend/ ./backend/

# Встановлюємо PYTHONPATH для коректного імпорту app.*
ENV PYTHONPATH="/app/backend"

# Порт FastAPI
EXPOSE 8000

# Запуск Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

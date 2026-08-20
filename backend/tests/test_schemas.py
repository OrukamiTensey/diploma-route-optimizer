"""
Тести валідації для доменних моделей TD-VRPTW-P.

Покриває:
  - Успішну ініціалізацію валідного OptimizationRequest
  - Невалідні координати (latitude, longitude)
  - Інвертоване часове вікно (start_time > end_time)
  - Пріоритет поза допустимим діапазоном [1, 5]
  - Порожній список завдань
  - Граничні значення (edge cases)
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.models import Location, OptimizationRequest, Task, TimeWindow


# =====================================================================
# Фікстури (готові валідні об'єкти для повторного використання)
# =====================================================================


@pytest.fixture
def depot() -> Location:
    """Валідне депо (КПІ, Київ)."""
    return Location(
        id="depot-001",
        latitude=50.4488,
        longitude=30.4571,
        name="КПІ ім. Ігоря Сікорського",
    )


@pytest.fixture
def valid_time_window() -> TimeWindow:
    """Валідне часове вікно: 09:00–12:00 за UNIX-мітками."""
    return TimeWindow(start_time=32400, end_time=43200)  # 9h, 12h


@pytest.fixture
def valid_task(valid_time_window: TimeWindow) -> Task:
    """Валідне завдання з локацією, часовим вікном та пріоритетом 3."""
    return Task(
        id="task-001",
        location=Location(
            id="loc-client-1",
            latitude=50.4501,
            longitude=30.5234,
            name="Хрещатик",
        ),
        time_window=valid_time_window,
        service_duration=900,  # 15 хвилин
        priority=3,
    )


@pytest.fixture
def valid_request(depot: Location, valid_task: Task) -> OptimizationRequest:
    """Повністю валідний запит оптимізації."""
    return OptimizationRequest(
        depot=depot,
        tasks=[valid_task],
        start_time=28800,  # 08:00
        weights={"w1": 1.0, "w2": 10.0, "w3": 5.0},
    )


# =====================================================================
# Location — валідні сценарії
# =====================================================================


class TestLocationValid:
    """Тести успішного створення Location."""

    def test_basic_creation(self, depot: Location) -> None:
        assert depot.id == "depot-001"
        assert depot.latitude == 50.4488
        assert depot.longitude == 30.4571
        assert depot.name == "КПІ ім. Ігоря Сікорського"

    def test_without_name(self) -> None:
        loc = Location(id="loc-1", latitude=0.0, longitude=0.0)
        assert loc.name is None

    def test_boundary_coordinates(self) -> None:
        """Граничні значення: полюси та антимеридіан."""
        loc_north_pole = Location(id="np", latitude=90.0, longitude=0.0)
        loc_south_pole = Location(id="sp", latitude=-90.0, longitude=0.0)
        loc_antimeridian = Location(id="am", latitude=0.0, longitude=180.0)
        loc_antimeridian_neg = Location(id="am-", latitude=0.0, longitude=-180.0)

        assert loc_north_pole.latitude == 90.0
        assert loc_south_pole.latitude == -90.0
        assert loc_antimeridian.longitude == 180.0
        assert loc_antimeridian_neg.longitude == -180.0


# =====================================================================
# Location — невалідні сценарії
# =====================================================================


class TestLocationInvalid:
    """Тести відловлення помилок валідації Location."""

    def test_latitude_too_high(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Location(id="bad", latitude=90.1, longitude=0.0)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("latitude",) for e in errors)

    def test_latitude_too_low(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Location(id="bad", latitude=-90.1, longitude=0.0)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("latitude",) for e in errors)

    def test_longitude_too_high(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Location(id="bad", latitude=0.0, longitude=180.1)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("longitude",) for e in errors)

    def test_longitude_too_low(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Location(id="bad", latitude=0.0, longitude=-180.1)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("longitude",) for e in errors)

    def test_empty_id(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Location(id="", latitude=0.0, longitude=0.0)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)


# =====================================================================
# TimeWindow — валідні сценарії
# =====================================================================


class TestTimeWindowValid:
    """Тести успішного створення TimeWindow."""

    def test_int_timestamps(self) -> None:
        tw = TimeWindow(start_time=100, end_time=200)
        assert tw.start_time == 100
        assert tw.end_time == 200

    def test_datetime_values(self) -> None:
        start = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
        tw = TimeWindow(start_time=start, end_time=end)
        assert tw.start_time == start
        assert tw.end_time == end

    def test_equal_start_end(self) -> None:
        """start_time == end_time дозволено (нульове вікно)."""
        tw = TimeWindow(start_time=500, end_time=500)
        assert tw.start_time == tw.end_time


# =====================================================================
# TimeWindow — невалідні сценарії
# =====================================================================


class TestTimeWindowInvalid:
    """Тести відловлення помилок валідації TimeWindow."""

    def test_inverted_int_window(self) -> None:
        """start_time > end_time має викликати помилку."""
        with pytest.raises(ValidationError) as exc_info:
            TimeWindow(start_time=200, end_time=100)
        assert "інвертоване" in str(exc_info.value).lower() or "start_time" in str(
            exc_info.value
        )

    def test_inverted_datetime_window(self) -> None:
        """Інвертоване часове вікно з datetime."""
        start = datetime(2026, 8, 20, 15, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
        with pytest.raises(ValidationError):
            TimeWindow(start_time=start, end_time=end)

    def test_mixed_types_rejected(self) -> None:
        """start_time (int) та end_time (datetime) — несумісні типи."""
        with pytest.raises(ValidationError) as exc_info:
            TimeWindow(
                start_time=100,
                end_time=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
            )
        assert "одного типу" in str(exc_info.value) or "типу" in str(exc_info.value)


# =====================================================================
# Task — валідні сценарії
# =====================================================================


class TestTaskValid:
    """Тести успішного створення Task."""

    def test_basic_creation(self, valid_task: Task) -> None:
        assert valid_task.id == "task-001"
        assert valid_task.location.id == "loc-client-1"
        assert valid_task.service_duration == 900
        assert valid_task.priority == 3

    def test_min_priority(self) -> None:
        task = Task(
            id="t-min",
            location=Location(id="l1", latitude=0.0, longitude=0.0),
            time_window=TimeWindow(start_time=0, end_time=100),
            service_duration=1,
            priority=1,
        )
        assert task.priority == 1

    def test_max_priority(self) -> None:
        task = Task(
            id="t-max",
            location=Location(id="l1", latitude=0.0, longitude=0.0),
            time_window=TimeWindow(start_time=0, end_time=100),
            service_duration=1,
            priority=5,
        )
        assert task.priority == 5


# =====================================================================
# Task — невалідні сценарії
# =====================================================================


class TestTaskInvalid:
    """Тести відловлення помилок валідації Task."""

    def test_priority_too_low(self) -> None:
        """Пріоритет 0 — поза допустимим діапазоном [1, 5]."""
        with pytest.raises(ValidationError) as exc_info:
            Task(
                id="bad-task",
                location=Location(id="l", latitude=0.0, longitude=0.0),
                time_window=TimeWindow(start_time=0, end_time=100),
                service_duration=60,
                priority=0,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("priority",) for e in errors)

    def test_priority_too_high(self) -> None:
        """Пріоритет 6 — поза допустимим діапазоном [1, 5]."""
        with pytest.raises(ValidationError) as exc_info:
            Task(
                id="bad-task",
                location=Location(id="l", latitude=0.0, longitude=0.0),
                time_window=TimeWindow(start_time=0, end_time=100),
                service_duration=60,
                priority=6,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("priority",) for e in errors)

    def test_negative_priority(self) -> None:
        """Від'ємний пріоритет."""
        with pytest.raises(ValidationError):
            Task(
                id="bad-task",
                location=Location(id="l", latitude=0.0, longitude=0.0),
                time_window=TimeWindow(start_time=0, end_time=100),
                service_duration=60,
                priority=-1,
            )

    def test_zero_service_duration(self) -> None:
        """service_duration = 0 не дозволено (має бути > 0)."""
        with pytest.raises(ValidationError) as exc_info:
            Task(
                id="bad-task",
                location=Location(id="l", latitude=0.0, longitude=0.0),
                time_window=TimeWindow(start_time=0, end_time=100),
                service_duration=0,
                priority=3,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("service_duration",) for e in errors)

    def test_negative_service_duration(self) -> None:
        """Від'ємна тривалість обслуговування."""
        with pytest.raises(ValidationError):
            Task(
                id="bad-task",
                location=Location(id="l", latitude=0.0, longitude=0.0),
                time_window=TimeWindow(start_time=0, end_time=100),
                service_duration=-10,
                priority=3,
            )


# =====================================================================
# OptimizationRequest — валідні сценарії
# =====================================================================


class TestOptimizationRequestValid:
    """Тести успішного створення OptimizationRequest."""

    def test_full_request(self, valid_request: OptimizationRequest) -> None:
        """Повний валідний запит з усіма полями."""
        assert valid_request.depot.id == "depot-001"
        assert len(valid_request.tasks) == 1
        assert valid_request.start_time == 28800
        assert valid_request.weights == {"w1": 1.0, "w2": 10.0, "w3": 5.0}

    def test_without_weights(self, depot: Location, valid_task: Task) -> None:
        """Запит без вагових коефіцієнтів (використовуються за замовчуванням)."""
        req = OptimizationRequest(
            depot=depot,
            tasks=[valid_task],
            start_time=0,
        )
        assert req.weights is None

    def test_multiple_tasks(self, depot: Location) -> None:
        """Запит з кількома завданнями."""
        tasks = [
            Task(
                id=f"task-{i}",
                location=Location(id=f"loc-{i}", latitude=50.0 + i * 0.01, longitude=30.0),
                time_window=TimeWindow(start_time=i * 1000, end_time=(i + 1) * 1000),
                service_duration=600,
                priority=min(i + 1, 5),
            )
            for i in range(5)
        ]
        req = OptimizationRequest(
            depot=depot,
            tasks=tasks,
            start_time=0,
        )
        assert len(req.tasks) == 5

    def test_datetime_start_time(self, depot: Location, valid_task: Task) -> None:
        """start_time може бути datetime."""
        req = OptimizationRequest(
            depot=depot,
            tasks=[valid_task],
            start_time=datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc),
        )
        assert isinstance(req.start_time, datetime)


# =====================================================================
# OptimizationRequest — невалідні сценарії
# =====================================================================


class TestOptimizationRequestInvalid:
    """Тести відловлення помилок валідації OptimizationRequest."""

    def test_empty_tasks_list(self, depot: Location) -> None:
        """Порожній список завдань заборонено."""
        with pytest.raises(ValidationError) as exc_info:
            OptimizationRequest(
                depot=depot,
                tasks=[],
                start_time=0,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("tasks",) for e in errors)

    def test_invalid_nested_location(self, depot: Location) -> None:
        """Невалідна локація всередині Task повинна каскадно відхилятися."""
        with pytest.raises(ValidationError):
            OptimizationRequest(
                depot=depot,
                tasks=[
                    Task(
                        id="bad-nested",
                        location=Location(id="l", latitude=999.0, longitude=0.0),
                        time_window=TimeWindow(start_time=0, end_time=100),
                        service_duration=60,
                        priority=3,
                    )
                ],
                start_time=0,
            )

    def test_invalid_nested_time_window(self, depot: Location) -> None:
        """Інвертоване часове вікно в Task каскадно відхиляється."""
        with pytest.raises(ValidationError):
            OptimizationRequest(
                depot=depot,
                tasks=[
                    Task(
                        id="bad-tw",
                        location=Location(id="l", latitude=50.0, longitude=30.0),
                        time_window=TimeWindow(start_time=200, end_time=100),
                        service_duration=60,
                        priority=3,
                    )
                ],
                start_time=0,
            )

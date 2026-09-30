"""Deterministic workout-plan checks used by the Workout Coach."""

import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from .profile import Experience

Muscle = Literal[
    "chest", "back", "shoulders", "biceps", "triceps", "quads", "hamstrings",
    "glutes", "calves", "core", "forearms", "full_body", "cardio",
]
MAJOR_MUSCLES = ("chest", "back", "shoulders", "quads", "hamstrings", "glutes", "biceps", "triceps")
NOT_COUNTED = {"full_body", "cardio"}

# Weekly hard sets per muscle group for hypertrophy, by training age.
WEEKLY_SET_RANGE = {"beginner": (6, 12), "intermediate": (10, 18), "advanced": (12, 22)}

SECONDS_PER_REP = 3.5
SETUP_SECONDS_PER_EXERCISE = 30


class Exercise(BaseModel):
    name: str
    primary_muscle: Muscle
    secondary_muscles: list[Muscle] = Field(default_factory=list)
    sets: int = Field(ge=1, le=10)
    reps: str = Field(description='e.g. "8-10", "12", "30s", "AMRAP".')
    rest_seconds: int = Field(ge=0, le=600)
    duration_minutes: float | None = Field(
        default=None, gt=0, le=90,
        description="For cardio or other continuous work; replaces the sets x reps time estimate.",
    )


class WorkoutDay(BaseModel):
    day: str = Field(description='e.g. "Monday - Upper A".')
    exercises: list[Exercise] = Field(min_length=1)


class WorkoutPlanInput(BaseModel):
    days: list[WorkoutDay] = Field(min_length=1, max_length=7)
    experience: Experience
    session_minutes_available: int = Field(ge=10, le=240)
    warmup_minutes: int = Field(default=8, ge=0, le=30)


def _work_seconds(reps: str) -> float:
    """Estimate the working time of one set from its rep prescription."""
    text = reps.lower()
    if (m := re.search(r"(\d+)\s*(s|sec|secs|second|seconds)\b", text)):
        return float(m.group(1))
    if (m := re.search(r"(\d+)\s*(min|mins|minute|minutes)\b", text)):
        return float(m.group(1)) * 60
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    avg_reps = sum(numbers[:2]) / len(numbers[:2]) if numbers else 12
    return min(avg_reps * SECONDS_PER_REP, 90)


def session_minutes(exercises: list[Exercise], warmup_minutes: int) -> int:
    """Warm-up + sets x (work time + rest) + setup time per exercise."""
    seconds = warmup_minutes * 60
    for ex in exercises:
        if ex.duration_minutes is not None:
            seconds += ex.duration_minutes * 60
        else:
            seconds += ex.sets * (_work_seconds(ex.reps) + ex.rest_seconds)
        seconds += SETUP_SECONDS_PER_EXERCISE
    return round(seconds / 60)


def analyze_workout_plan(inp: WorkoutPlanInput) -> dict:
    low, high = WEEKLY_SET_RANGE[inp.experience]
    weekly_sets: dict[str, float] = defaultdict(float)
    days_trained: dict[str, int] = defaultdict(int)
    sessions = []
    warnings: list[str] = []

    for day in inp.days:
        sets_today: dict[str, float] = defaultdict(float)
        for ex in day.exercises:
            if ex.primary_muscle not in NOT_COUNTED:
                sets_today[ex.primary_muscle] += ex.sets
            for muscle in set(ex.secondary_muscles) - NOT_COUNTED - {ex.primary_muscle}:
                sets_today[muscle] += ex.sets * 0.5

        for muscle, sets in sets_today.items():
            weekly_sets[muscle] += sets
            if sets >= 2:
                days_trained[muscle] += 1

        minutes = session_minutes(day.exercises, inp.warmup_minutes)
        over = minutes - inp.session_minutes_available
        if over > 5:
            status = f"over by {over} min"
            warnings.append(f"{day.day} runs ~{minutes} min, over the {inp.session_minutes_available} min available.")
        elif minutes < 0.6 * inp.session_minutes_available:
            status = "well under the time available"
        else:
            status = "fits"
        sessions.append({"day": day.day, "estimated_minutes": minutes, "status": status})

    volume = {}
    for muscle in sorted(set(weekly_sets) | set(MAJOR_MUSCLES), key=_muscle_order):
        sets = weekly_sets.get(muscle, 0.0)
        entry: dict = {"weekly_sets": round(sets, 1), "days_per_week": days_trained.get(muscle, 0)}
        if muscle in MAJOR_MUSCLES:
            entry["target_range"] = f"{low}-{high}"
            if sets < low:
                entry["status"] = "below range"
                warnings.append(f"{muscle}: {sets:g} weekly sets, below the {low}-{high} range.")
            elif sets > high:
                entry["status"] = "above range"
                warnings.append(f"{muscle}: {sets:g} weekly sets, above the {low}-{high} range.")
            else:
                entry["status"] = "in range"
            if len(inp.days) >= 3 and 0 < days_trained.get(muscle, 0) < 2:
                warnings.append(f"{muscle} is trained only once a week; twice usually works better.")
        volume[muscle] = entry

    return {
        "sessions": sessions,
        "weekly_volume": volume,
        "counting_rule": "Primary muscle = 1 set, each secondary muscle = 0.5 set. Ranges are for hypertrophy.",
        "warnings": warnings,
    }


def _muscle_order(muscle: str) -> int:
    order = list(MAJOR_MUSCLES) + ["calves", "core", "forearms"]
    return order.index(muscle) if muscle in order else len(order)

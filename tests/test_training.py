from app.training import WorkoutPlanInput, _work_seconds, analyze_workout_plan


def ex(name, primary, sets, secondary=(), reps="8-10", rest=90, **kw):
    return {"name": name, "primary_muscle": primary, "secondary_muscles": list(secondary),
            "sets": sets, "reps": reps, "rest_seconds": rest, **kw}


def test_work_seconds_parsing():
    assert _work_seconds("8-10") == 9 * 3.5
    assert _work_seconds("30s") == 30
    assert _work_seconds("1 min") == 60
    assert _work_seconds("AMRAP") == 12 * 3.5


def test_session_duration_and_status():
    plan = WorkoutPlanInput.model_validate({
        "days": [{"day": "Mon", "exercises": [ex("Squat", "quads", 4, rest=150), ex("Bike", "cardio", 1, duration_minutes=10)]}],
        "experience": "beginner",
        "session_minutes_available": 30,
        "warmup_minutes": 5,
    })
    result = analyze_workout_plan(plan)
    # 5 min warm-up + 4 * (31.5 + 150) s + 10 min + 2 * 30 s setup = 1686 s
    assert result["sessions"][0]["estimated_minutes"] == 28
    assert result["sessions"][0]["status"] == "fits"


def test_over_time_is_flagged():
    plan = WorkoutPlanInput.model_validate({
        "days": [{"day": "Mon", "exercises": [ex(f"Lift {i}", "chest", 5, rest=180) for i in range(5)]}],
        "experience": "intermediate",
        "session_minutes_available": 45,
    })
    result = analyze_workout_plan(plan)
    assert result["sessions"][0]["status"].startswith("over by")
    assert any("over the 45 min" in w for w in result["warnings"])


def test_weekly_volume_counts_secondary_as_half():
    day = [ex("Bench", "chest", 4, secondary=["triceps", "shoulders"]), ex("Row", "back", 4, secondary=["biceps"])]
    plan = WorkoutPlanInput.model_validate({
        "days": [{"day": "A", "exercises": day}, {"day": "B", "exercises": day}],
        "experience": "beginner",
        "session_minutes_available": 60,
    })
    volume = analyze_workout_plan(plan)["weekly_volume"]
    assert volume["chest"] == {"weekly_sets": 8, "days_per_week": 2, "target_range": "6-12", "status": "in range"}
    assert volume["triceps"]["weekly_sets"] == 4
    assert volume["quads"]["status"] == "below range"

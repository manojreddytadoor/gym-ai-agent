"""Coach flow tests with a scripted stand-in for the local model."""

import pytest

from app.coaches import DietCoach, WorkoutCoach
from app.exercises import EXERCISES
from app.llm import LLMError
from app.nutrition import day_totals
from app.training import session_minutes
from app.profile import UserProfile


class FakeLLM:
    model = "fake"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[list[dict], dict]] = []

    def generate_json(self, messages, schema, *, status, caller):
        self.calls.append((messages, schema))
        yield {"type": "status", "text": status}
        yield {"type": "progress", "tokens": 10}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def profile(**overrides):
    base = dict(
        age=28, sex="male", height_cm=175, weight_kg=82, goal="fat_loss", daily_activity="sedentary",
        experience="beginner", training_days_per_week=4, session_minutes=60, workout_time="evening",
        equipment="full_gym", diet_type="non_vegetarian", meals_per_day=4,
    )
    return UserProfile(**(base | overrides))


def meal(name, *items):
    return {"name": name, "time": "8:00 am", "dish": f"{name} dish",
            "items": [{"food": f, "grams": g} for f, g in items]}


GOOD_MEALS = {
    "meals": [
        meal("Breakfast", ("oats (dry)", 60), ("milk (whole)", 250), ("banana", 120), ("whey protein powder", 30)),
        meal("Lunch", ("chicken breast (cooked)", 150), ("white rice (cooked)", 200),
             ("mixed vegetables (cooked)", 150), ("cooking oil", 10)),
        meal("Snack", ("greek yogurt (low-fat)", 200), ("apple", 180), ("almonds", 15)),
        meal("Dinner", ("white fish (cooked)", 150), ("roti / chapati", 80), ("spinach", 100), ("cooking oil", 5)),
    ],
    "swaps": ["Swap chicken for fish"],
    "tips": ["Eat protein at every meal"],
}
NO_PROTEIN_MEALS = {
    "meals": [meal(n, ("white rice (cooked)", 200), ("banana", 120)) for n in ("Breakfast", "Lunch", "Snack", "Dinner")],
    "swaps": [], "tips": [],
}


def run(coach, message=None):
    return list(coach.chat(message))


def test_diet_plan_is_fitted_to_targets():
    llm = FakeLLM(GOOD_MEALS)
    coach = DietCoach(llm, profile())

    events = run(coach)

    reply = next(e for e in events if e["type"] == "reply")["markdown"]
    assert events[-1] == {"type": "done"}
    assert "✅ Calories are within 5%" in reply
    day = day_totals(coach.plan["meals"])
    assert day["kcal"] == pytest.approx(coach.targets["target_kcal"], rel=0.05)
    assert day["protein_g"] >= 0.95 * coach.targets["protein_g"]
    assert len(llm.calls) == 1
    meals_schema = llm.calls[0][1]["properties"]["meals"]
    assert meals_schema["minItems"] == meals_schema["maxItems"] == 4


def test_dish_names_lose_quantities_and_excluded_foods():
    raw = {"meals": [m | {} for m in GOOD_MEALS["meals"]], "swaps": [], "tips": []}
    raw["meals"][0] = raw["meals"][0] | {"dish": "Paneer bhurji with 2 rotis"}
    raw["meals"][1] = raw["meals"][1] | {"dish": "Chicken curry with 150g rice"}
    plan, _, _ = DietCoach(FakeLLM(), profile()).check(raw)
    assert plan["meals"][0].dish == "Paneer bhurji with rotis"
    assert plan["meals"][1].dish == "Chicken curry with rice"

    veg_plan, _, _ = DietCoach(FakeLLM(), profile(diet_type="vegetarian")).check(raw)
    assert veg_plan["meals"][1].dish == ""


def test_diet_schema_only_offers_allowed_foods():
    schema = DietCoach(FakeLLM(), profile(diet_type="vegetarian", allergies="peanuts")).schema(follow_up=False)
    enum = schema["properties"]["meals"]["items"]["properties"]["items"]["items"]["properties"]["food"]["enum"]
    assert "paneer" in enum
    assert not {"chicken breast (cooked)", "egg (whole)", "peanuts", "peanut butter"} & set(enum)


def test_diet_asks_for_revision_when_protein_cannot_be_reached():
    llm = FakeLLM(NO_PROTEIN_MEALS, GOOD_MEALS)
    coach = DietCoach(llm, profile())

    events = run(coach)

    assert len(llm.calls) == 2
    feedback = llm.calls[1][0][-1]["content"]
    assert "Protein reaches only" in feedback and "such as prawns (cooked)" in feedback
    assert {"type": "status", "text": "Revising the plan to fix 1 problem the checks found"} in events
    assert "✅" in next(e for e in events if e["type"] == "reply")["markdown"]


def test_diet_gives_up_after_max_revisions_and_says_so():
    llm = FakeLLM(NO_PROTEIN_MEALS, NO_PROTEIN_MEALS, NO_PROTEIN_MEALS)
    events = run(DietCoach(llm, profile()))
    reply = next(e for e in events if e["type"] == "reply")["markdown"]
    assert "Couldn't fully fix" in reply and "⚠️" in reply


def test_diet_follow_up_question_keeps_plan():
    llm = FakeLLM(GOOD_MEALS, {"reply": "Protein helps you keep muscle.", "update_plan": False, "meals": []})
    coach = DietCoach(llm, profile())
    run(coach)
    plan_before = coach.plan

    events = run(coach, "Why so much protein?")

    assert {"type": "reply", "markdown": "Protein helps you keep muscle."} in events
    assert coach.plan is plan_before
    assert "Why so much protein?" in llm.calls[1][0][-1]["content"]


def test_diet_follow_up_change_updates_plan():
    updated = GOOD_MEALS | {"reply": "Swapped the fish for chicken.", "update_plan": True, "swaps": [], "tips": []}
    updated["meals"] = [m | {} for m in GOOD_MEALS["meals"]]
    updated["meals"][3] = meal("Dinner", ("chicken thigh (cooked, skinless)", 150), ("roti / chapati", 80))
    llm = FakeLLM(GOOD_MEALS, updated)
    coach = DietCoach(llm, profile())
    run(coach)

    events = run(coach, "No fish please")

    reply = next(e for e in events if e["type"] == "reply")["markdown"]
    assert reply.startswith("Swapped the fish for chicken.")
    assert coach.plan["meals"][3].items[0].food == "chicken thigh (cooked, skinless)"
    assert coach.plan["swaps"] == ["Swap chicken for fish"]  # kept from before


def test_model_error_is_reported_and_plan_unchanged():
    coach = DietCoach(FakeLLM(LLMError("Can't connect to Ollama.")), profile())
    events = run(coach)
    assert events[-1] == {"type": "error", "message": "Can't connect to Ollama."}
    assert not any(e["type"] in ("reply", "done") for e in events)
    assert coach.plan is None




def split(name):
    return {"split": name}


def picks(*pairs):
    return {"exercises": [{"exercise": n, "sets": s} for n, s in pairs]}


UPPER = picks(("Barbell bench press", 3), ("Lat pulldown", 3), ("Seated dumbbell shoulder press", 3),
              ("Seated cable row", 3), ("Dumbbell curl", 2), ("Triceps pushdown", 2))
LOWER = picks(("Barbell back squat", 3), ("Barbell hip thrust", 3), ("Leg press", 3),
              ("Lying leg curl", 3), ("Standing calf raise", 3), ("Plank", 2))
UPPER_LOWER_WEEK = [split("Upper / Lower"), UPPER, LOWER, UPPER, LOWER]
UPPER_LOWER_FOCUS = ["Upper body A", "Lower body A", "Upper body B", "Lower body B"]


def test_workout_builds_the_week_one_day_at_a_time():
    llm = FakeLLM(*UPPER_LOWER_WEEK)
    coach = WorkoutCoach(llm, profile())

    events = run(coach)

    assert len(llm.calls) == 5  # one split call, then one per training day
    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert statuses[0] == "Choosing your training split"
    assert statuses[1:] == [f"Building day {i} of 4: {f}" for i, f in enumerate(UPPER_LOWER_FOCUS, start=1)]
    assert [d.weekday for d in coach.plan["days"]] == ["Monday", "Tuesday", "Thursday", "Friday"]

    reply = next(e for e in events if e["type"] == "reply")["markdown"]
    assert "**Upper / Lower** · 4 days a week (Mon, Tue, Thu, Fri)" in reply
    assert "### Monday: Upper body A · about 49 min" in reply
    assert "⬇️" not in reply  # every major muscle in range


def test_app_fills_in_reps_rest_and_effort():
    coach = WorkoutCoach(FakeLLM(), profile(goal="strength", experience="advanced"))
    day = coach._build_day("Monday", "Upper", UPPER["exercises"])
    bench = day.exercises[0]
    assert bench.name == "Barbell bench press"
    assert (bench.reps_low, bench.reps_high) == (4, 6)  # strength range for a compound lift
    assert bench.rest_seconds == 180
    assert bench.rir == 2  # advanced 1, plus 1 for a heavy compound

    curl = next(e for e in day.exercises if e.name == "Dumbbell curl")
    assert (curl.reps_low, curl.reps_high, curl.rest_seconds, curl.rir) == (8, 12, 75, 1)


def test_sessions_are_trimmed_to_the_time_available():
    coach = WorkoutCoach(FakeLLM(), profile(session_minutes=30))
    assert coach.max_exercises == 3  # a 30 minute session only fits three exercises

    day = coach._build_day("Monday", "Full body", [
        {"exercise": "Barbell back squat", "sets": 5},
        {"exercise": "Barbell bench press", "sets": 5},
        {"exercise": "Seated cable row", "sets": 5},
    ])
    # Sets are trimmed from the end, so the main lift of the day keeps its volume.
    assert [e.sets for e in day.exercises] == [5, 2, 2]
    assert session_minutes([e.timed() for e in day.exercises], 8) <= 33


def test_compound_lifts_come_first_and_cardio_last():
    coach = WorkoutCoach(FakeLLM(), profile())
    day = coach._build_day("Monday", "Full body", [
        {"exercise": "Stationary bike", "sets": 1},
        {"exercise": "Dumbbell curl", "sets": 2},
        {"exercise": "Barbell back squat", "sets": 3},
    ])
    assert [e.name for e in day.exercises] == ["Barbell back squat", "Dumbbell curl", "Stationary bike"]
    assert day.warmup.endswith("ramp-up sets of barbell back squat.")


def test_too_much_weekly_volume_is_trimmed_automatically():
    coach = WorkoutCoach(FakeLLM(), profile(training_days_per_week=5))
    days = [coach._build_day(w, "Upper body", UPPER["exercises"])
            for w in ["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"]]
    before = coach._analysis(days)["weekly_volume"]

    coach._trim_overshoot(days)

    after = coach._analysis(days)["weekly_volume"]
    assert before["chest"]["weekly_sets"] > 12  # five upper days is far too much for a beginner
    assert after["chest"]["weekly_sets"] <= 12
    assert all(after[m]["status"] != "above range" for m in ("chest", "back", "biceps", "triceps"))
    assert all(e.sets >= 2 for d in days for e in d.exercises)  # never trimmed below two sets


def test_a_day_that_misses_its_focus_muscles_is_rebuilt():
    # The model hands back an upper-body day when a leg day was asked for.
    llm = FakeLLM(split("Upper / Lower"), UPPER, UPPER, LOWER, UPPER, LOWER)
    coach = WorkoutCoach(llm, profile())

    events = run(coach)

    assert len(llm.calls) == 6  # split, four days, one rebuild of Tuesday
    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert "Adding quads, hamstrings, glutes to Tuesday" in statuses
    assert "must include one for each of: quads, hamstrings, glutes" in llm.calls[3][0][-1]["content"]
    assert "Barbell back squat" in [e.name for e in coach.plan["days"][1].exercises]


def test_the_app_adds_the_muscle_itself_if_the_model_keeps_missing_it():
    llm = FakeLLM(split("Upper / Lower"), UPPER, UPPER, UPPER, UPPER, UPPER, UPPER)
    coach = WorkoutCoach(llm, profile())

    run(coach)

    tuesday = coach.plan["days"][1]  # asked for Lower body, model only ever gave upper work
    assert {EXERCISES[e.name].primary for e in tuesday.exercises} >= {"quads", "hamstrings", "glutes"}
    assert session_minutes([e.timed() for e in tuesday.exercises], 8) <= coach.profile.session_minutes + 3


def test_weekly_volume_gaps_trigger_a_rebuild():
    # Two full-body days that satisfy each day's focus but leave the week too light.
    thin = picks(("Barbell bench press", 2), ("Seated cable row", 2), ("Barbell back squat", 2),
                 ("Dumbbell curl", 2))
    full = picks(("Barbell bench press", 4), ("Seated cable row", 4), ("Barbell back squat", 4),
                 ("Dumbbell lateral raise", 3), ("Lying leg curl", 3), ("Barbell hip thrust", 3))
    llm = FakeLLM(thin, thin, full, full)  # two days a week has one split template, so no split call
    coach = WorkoutCoach(llm, profile(training_days_per_week=2))

    events = run(coach)

    assert [e["text"] for e in events if e["type"] == "status" and e["text"].startswith("Adding")]
    assert len(llm.calls) > 2  # the balancer asked for at least one rebuild


def test_follow_up_rebuilds_one_day():
    llm = FakeLLM(*UPPER_LOWER_WEEK, {
        "reply": "Swapped Friday to a shorter session.", "change_day": "Friday",
        "exercises": [{"exercise": "Goblet squat", "sets": 3}, {"exercise": "Lying leg curl", "sets": 3},
                      {"exercise": "Standing calf raise", "sets": 2}],
    })
    coach = WorkoutCoach(llm, profile())
    run(coach)

    events = run(coach, "Friday needs to be shorter")

    reply = next(e for e in events if e["type"] == "reply")["markdown"]
    assert reply.startswith("Swapped Friday to a shorter session.")
    assert [e.name for e in coach.plan["days"][3].exercises] == ["Goblet squat", "Lying leg curl",
                                                                "Standing calf raise"]
    assert [d.weekday for d in coach.plan["days"]] == ["Monday", "Tuesday", "Thursday", "Friday"]


def test_follow_up_question_leaves_the_plan_alone():
    llm = FakeLLM(*UPPER_LOWER_WEEK,
                  {"reply": "RIR means reps in reserve.", "change_day": "none", "exercises": []})
    coach = WorkoutCoach(llm, profile())
    run(coach)
    before = [e.name for e in coach.plan["days"][0].exercises]

    events = run(coach, "What does RIR mean?")

    assert {"type": "reply", "markdown": "RIR means reps in reserve."} in events
    assert [e.name for e in coach.plan["days"][0].exercises] == before


def test_workout_schema_respects_equipment_and_injuries():
    coach = WorkoutCoach(FakeLLM(), profile(equipment="dumbbells_only", injuries="bad knee"))
    enum = coach._day_schema()["properties"]["exercises"]["items"]["properties"]["exercise"]["enum"]
    assert "Dumbbell bench press" in enum
    assert not {"Barbell bench press", "Goblet squat", "Lat pulldown"} & set(enum)


def test_exercises_outside_the_list_are_dropped():
    coach = WorkoutCoach(FakeLLM(), profile(equipment="bodyweight_only"))
    day = coach._build_day("Monday", "Full body", [
        {"exercise": "Barbell back squat", "sets": 3},  # no barbell at home
        {"exercise": "Push-up", "sets": 3},
        {"exercise": "Push-up", "sets": 2},  # duplicate
    ])
    assert [e.name for e in day.exercises] == ["Push-up"]

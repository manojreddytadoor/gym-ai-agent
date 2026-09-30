import typing

from app.exercises import EXERCISES, allowed_exercises, format_amount, injury_tags
from app.training import MAJOR_MUSCLES, Muscle

MUSCLES = set(typing.get_args(Muscle))


def test_database_uses_known_muscles_and_units():
    for name, ex in EXERCISES.items():
        assert ex.primary in MUSCLES, name
        assert set(ex.secondary) <= MUSCLES, name
        assert ex.unit in {"reps", "seconds", "minutes"}, name
        assert ex.cue, name


def test_bodyweight_only_covers_major_muscles_except_biceps():
    allowed = allowed_exercises("bodyweight_only", "")
    assert all(EXERCISES[n].equipment == {"bodyweight"} for n in allowed)
    # Biceps have no good bodyweight exercise without a bar; rows still train them indirectly.
    assert set(MAJOR_MUSCLES) - {"biceps"} <= {EXERCISES[n].primary for n in allowed}
    for muscle in MAJOR_MUSCLES:
        assert muscle in {EXERCISES[n].primary for n in allowed_exercises("full_gym", "")}


def test_equipment_filtering():
    dumbbells = allowed_exercises("dumbbells_only", "")
    assert "Goblet squat" in dumbbells
    assert "Barbell back squat" not in dumbbells and "Lat pulldown" not in dumbbells and "Pull-up" not in dumbbells
    assert "Pull-up" in allowed_exercises("home_gym", "")


def test_injuries_remove_exercises_that_load_the_joint():
    assert injury_tags("mild lower-back pain, old left-knee injury") == {"lower_back", "knee"}
    allowed = allowed_exercises("full_gym", "mild lower-back pain")
    assert "Barbell deadlift" not in allowed and "Barbell back squat" not in allowed
    assert "Leg press" in allowed and "Glute bridge" in allowed
    assert "Leg press" not in allowed_exercises("full_gym", "torn meniscus")
    assert len(allowed_exercises("full_gym", "none")) == len(EXERCISES)


def test_format_amount():
    assert format_amount("Goblet squat", 8, 12) == "8-12"
    assert format_amount("Plank", 30, 45) == "30-45 s"
    assert format_amount("Stationary bike", 20, 20) == "20 min"

import pytest

from app.nutrition import (
    FOODS,
    NutritionTargetsInput,
    calculate_nutrition_targets,
    day_totals,
    fit_portions,
    food_rules,
    household_measure,
    normalize_meals,
)


def targets(**overrides):
    base = dict(
        weight_kg=80, height_cm=180, age=30, sex="male", daily_activity="sedentary",
        training_days_per_week=4, session_minutes=60, goal="general_fitness",
    )
    return calculate_nutrition_targets(NutritionTargetsInput(**(base | overrides)))


def test_mifflin_st_jeor_bmr():
    # 10*80 + 6.25*180 - 5*30 + 5 = 1780
    assert targets()["bmr_kcal"] == 1780
    # 10*60 + 6.25*165 - 5*25 - 161 = 1345.25
    assert targets(weight_kg=60, height_cm=165, age=25, sex="female")["bmr_kcal"] == 1345


def test_activity_and_goal_adjustments():
    t = targets()  # sedentary 1.2 + 4 days * 1h * 0.05 = 1.4
    assert t["activity_factor"] == 1.4
    assert t["maintenance_kcal"] == 2490  # 1780 * 1.4 = 2492
    assert targets(goal="fat_loss")["target_kcal"] == 1990  # 2492 * 0.8
    assert targets(goal="muscle_gain")["target_kcal"] == 2740  # 2492 * 1.1


def test_macros_add_up_to_target():
    t = targets(goal="fat_loss")
    kcal_from_macros = t["protein_g"] * 4 + t["carbs_g"] * 4 + t["fat_g"] * 9
    assert kcal_from_macros == pytest.approx(t["target_kcal"], abs=20)
    assert t["protein_g"] == 160  # 2.0 g/kg


def test_calorie_floor_applies():
    t = targets(weight_kg=45, height_cm=150, age=60, sex="female", goal="fat_loss", training_days_per_week=0)
    assert t["target_kcal"] >= 1200
    assert any("floor" in note for note in t["notes"])


def test_high_bmi_uses_reference_weight_for_protein():
    t = targets(weight_kg=130, height_cm=175, goal="fat_loss")
    assert t["protein_g"] == round(2.0 * 25 * 1.75**2)


def test_under_18_deficit_is_limited():
    t = targets(age=17, goal="fat_loss")
    assert t["target_kcal"] == 2320  # BMR 1845 * 1.4 * 0.9, not * 0.8
    assert any("Under 18" in note for note in t["notes"])


def test_food_database_values_are_consistent():
    # Atwater check: kcal ~ 4*protein + 4*net carbs + 2*fiber + 9*fat
    for name, f in FOODS.items():
        estimate = 4 * f.protein + 4 * (f.carbs - f.fiber) + 2 * f.fiber + 9 * f.fat
        assert estimate == pytest.approx(f.kcal, rel=0.2), name
        assert f.max_g >= (f.unit[1] if f.unit and f.unit[2] else 5), name


def test_diet_types_exclude_the_right_foods():
    veg = food_rules("vegetarian", "").allowed
    assert "paneer" in veg and "chicken breast (cooked)" not in veg and "egg (whole)" not in veg
    egg = food_rules("eggetarian", "").allowed
    assert "egg (whole)" in egg and "white fish (cooked)" not in egg
    vegan = food_rules("vegan", "").allowed
    assert not {"paneer", "milk (whole)", "ghee", "honey", "whey protein powder"} & set(vegan)
    assert "tofu (firm)" in vegan and "soy milk (unsweetened)" in vegan


@pytest.mark.parametrize("allergies, excluded, kept", [
    ("peanuts", {"peanuts", "peanut butter"}, {"almonds"}),
    ("lactose intolerant", {"milk (whole)", "paneer", "ghee"}, {"soy milk (unsweetened)"}),
    ("tree nuts", {"almonds", "cashews", "peanuts"}, {"chia seeds"}),
    ("I don't eat mushrooms, and no rice", {"mushrooms", "white rice (cooked)", "brown rice (cooked)"}, {"potato (boiled)"}),
    ("gluten", {"roti / chapati", "oats (dry)", "pasta (cooked)"}, {"white rice (cooked)"}),
    ("none", set(), set(FOODS)),
])
def test_allergies_exclude_foods(allergies, excluded, kept):
    allowed = set(food_rules("non_vegetarian", allergies).allowed)
    assert not excluded & allowed
    assert kept <= allowed


def test_mentions_excluded_in_free_text():
    veg = food_rules("vegetarian", "")
    assert veg.mentions_excluded("Egg bhurji with toast")
    assert not veg.mentions_excluded("Baingan (eggplant) bharta")
    no_dairy = food_rules("vegan", "")
    assert no_dairy.mentions_excluded("Swap curd for yogurt")
    assert not no_dairy.mentions_excluded("Peanut butter toast with soy milk")


def test_normalize_merges_duplicates_and_clamps_portions():
    meals = normalize_meals([{"name": "Lunch", "time": "1 pm", "dish": "", "items": [
        {"food": "curd (whole milk)", "grams": 200},
        {"food": "curd (whole milk)", "grams": 100},  # duplicate: merged
        {"food": "oats (dry)", "grams": 250},  # above the 100 g limit
        {"food": "roti / chapati", "grams": 70},  # rounded to whole rotis
        {"food": "chicken breast (cooked)", "grams": 100},  # not allowed: dropped
    ]}], allowed=food_rules("vegetarian", "").allowed)
    assert [(i.food, i.grams) for i in meals[0].items] == [
        ("curd (whole milk)", 300), ("oats (dry)", 100), ("roti / chapati", 80),
    ]


def test_fit_portions_repairs_small_model_output():
    # What llama3.2 actually produced for "~2000 kcal, high protein, vegetarian".
    raw = [
        {"name": "Breakfast", "items": [{"food": "oats (dry)", "grams": 250}, {"food": "almonds", "grams": 30},
                                        {"food": "milk (whole)", "grams": 200}, {"food": "banana", "grams": 100}]},
        {"name": "Lunch", "items": [{"food": "curd (whole milk)", "grams": 200}, {"food": "paneer", "grams": 150},
                                    {"food": "spinach", "grams": 100}, {"food": "curd (whole milk)", "grams": 100}]},
        {"name": "Dinner", "items": [{"food": "lentils / dal (cooked)", "grams": 250}, {"food": "milk (whole)", "grams": 200},
                                     {"food": "almonds", "grams": 30}, {"food": "curd (whole milk)", "grams": 100}]},
    ]
    goal = {"target_kcal": 2000, "protein_g": 100, "fat_g": 60}
    meals = normalize_meals(raw, food_rules("vegetarian", "").allowed)
    fit_portions(meals, goal)
    day = day_totals(meals)
    assert day["kcal"] == pytest.approx(2000, rel=0.05)
    assert day["protein_g"] >= 95
    for meal in meals:
        for item in meal.items:
            assert item.grams <= FOODS[item.food].max_g


def test_household_measures():
    assert household_measure("roti / chapati", 80) == "2 rotis"
    assert household_measure("egg (whole)", 50) == "1 egg"
    assert household_measure("cooking oil", 10) == "about 2 tsp"
    assert household_measure("milk (whole)", 250) == "about 1 glass"
    assert household_measure("brown rice (cooked)", 110) == "about ¾ cup"
    assert household_measure("white rice (cooked)", 240) == "about 1½ cups"
    assert household_measure("paneer", 100) is None

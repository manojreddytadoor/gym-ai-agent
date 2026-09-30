"""Nutrition math, the food database, and portion fitting for the Diet Coach.

The language model only chooses foods (from a filtered list) and rough portions.
Every calorie and macro number shown to the user is calculated here.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .profile import DailyActivity, DietType, Goal, Sex

# ---------------------------------------------------------------------------
# Calorie and macro targets
# ---------------------------------------------------------------------------

ACTIVITY_BASE = {"sedentary": 1.2, "lightly_active": 1.35, "active": 1.5}
GOAL_CALORIE_FACTOR = {
    "fat_loss": 0.80,
    "muscle_gain": 1.10,
    "recomposition": 0.95,
    "strength": 1.05,
    "general_fitness": 1.00,
    "endurance": 1.00,
}
PROTEIN_G_PER_KG = {
    "fat_loss": 2.0,
    "muscle_gain": 1.8,
    "recomposition": 2.0,
    "strength": 1.8,
    "general_fitness": 1.4,
    "endurance": 1.5,
}
CALORIE_FLOOR = {"male": 1500, "female": 1200, "other": 1350}
KCAL_PER_KG_BODY_FAT = 7700


class NutritionTargetsInput(BaseModel):
    weight_kg: float = Field(ge=35, le=250)
    height_cm: float = Field(ge=120, le=230)
    age: int = Field(ge=16, le=90)
    sex: Sex
    daily_activity: DailyActivity
    training_days_per_week: int = Field(ge=0, le=7)
    session_minutes: int = Field(ge=0, le=180)
    goal: Goal


def calculate_nutrition_targets(inp: NutritionTargetsInput) -> dict:
    """Mifflin-St Jeor BMR -> TDEE -> goal-adjusted calories and macros."""
    sex_offset = {"male": 5, "female": -161, "other": -78}[inp.sex]
    bmr = 10 * inp.weight_kg + 6.25 * inp.height_cm - 5 * inp.age + sex_offset

    training_load = inp.training_days_per_week * min(inp.session_minutes / 60, 1.5)
    activity_factor = min(ACTIVITY_BASE[inp.daily_activity] + 0.05 * training_load, 1.9)
    tdee = bmr * activity_factor

    notes: list[str] = []
    goal_factor = GOAL_CALORIE_FACTOR[inp.goal]
    if inp.age < 18 and goal_factor < 0.9:
        goal_factor = 0.9
        notes.append("Under 18: deficit limited to 10% to protect growth and development.")
    target = tdee * goal_factor

    floor = max(bmr, CALORIE_FLOOR[inp.sex])
    if target < floor:
        target = floor
        notes.append(f"Calorie target raised to the safety floor of {round(floor)} kcal.")

    height_m = inp.height_cm / 100
    bmi = inp.weight_kg / height_m**2
    # For higher BMIs, base protein and fat minimums on the weight at BMI 25 so
    # targets aren't inflated by body fat.
    reference_weight = min(inp.weight_kg, 25 * height_m**2) if bmi >= 30 else inp.weight_kg
    if reference_weight != inp.weight_kg:
        notes.append(f"Protein based on a reference weight of {reference_weight:.0f} kg (BMI 25) because BMI is 30+.")

    protein_g = PROTEIN_G_PER_KG[inp.goal] * reference_weight
    fat_g = max(0.25 * target / 9, 0.6 * reference_weight)
    carbs_g = max((target - protein_g * 4 - fat_g * 9) / 4, 0)

    weekly_change_kg = (target - tdee) * 7 / KCAL_PER_KG_BODY_FAT
    water_rest = 0.035 * inp.weight_kg
    water_training = water_rest + 0.5 * max(inp.session_minutes, 0) / 60

    return {
        "bmr_kcal": round(bmr),
        "activity_factor": round(activity_factor, 2),
        "maintenance_kcal": _round10(tdee),
        "target_kcal": _round10(target),
        "protein_g": round(protein_g),
        "carbs_g": round(carbs_g),
        "fat_g": round(fat_g),
        "fiber_g": round(14 * target / 1000),
        "water_liters_rest_day": round(water_rest, 1),
        "water_liters_training_day": round(water_training, 1),
        "bmi": round(bmi, 1),
        "bmi_category": _bmi_category(bmi),
        "expected_weekly_weight_change_kg": round(weekly_change_kg, 2),
        "notes": notes,
    }


def _round10(x: float) -> int:
    return int(round(x / 10) * 10)


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


# ---------------------------------------------------------------------------
# Food database
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Food:
    """Nutrition per 100 g, plus what the portion fitter needs to know."""

    kcal: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    group: str  # protein | dairy | legume | grain | fruit | vegetable | fat | sweet
    tags: frozenset[str] = field(default_factory=frozenset)  # diet and allergen tags
    max_g: int = 300  # largest sensible portion in one meal
    unit: tuple[str, int, bool] | None = None  # (name, grams, countable) for household measures

    @property
    def protein_share(self) -> float:
        return self.protein * 4 / self.kcal if self.kcal else 0.0

    @property
    def fat_share(self) -> float:
        return self.fat * 9 / self.kcal if self.kcal else 0.0


def _f(kcal, protein, carbs, fat, fiber, group, tags="", max_g=300, unit=None) -> Food:
    return Food(kcal, protein, carbs, fat, fiber, group, frozenset(tags.split()), max_g, unit)


# Approximate values from USDA FoodData Central / IFCT; brands and recipes vary.
FOODS: dict[str, Food] = {
    # Protein foods
    "chicken breast (cooked)": _f(165, 31.0, 0.0, 3.6, 0.0, "protein", "meat", 250),
    "chicken thigh (cooked, skinless)": _f(209, 26.0, 0.0, 10.9, 0.0, "protein", "meat", 250),
    "egg (whole)": _f(143, 12.6, 0.7, 9.5, 0.0, "protein", "egg", 200, ("egg", 50, True)),
    "egg white": _f(52, 10.9, 0.7, 0.2, 0.0, "protein", "egg", 264, ("egg white", 33, True)),
    "white fish (cooked)": _f(128, 26.0, 0.0, 2.7, 0.0, "protein", "fish", 250),
    "salmon (cooked)": _f(206, 22.1, 0.0, 12.4, 0.0, "protein", "fish", 200),
    "tuna (canned in water)": _f(116, 25.5, 0.0, 0.8, 0.0, "protein", "fish", 150),
    "prawns (cooked)": _f(99, 24.0, 0.2, 0.3, 0.0, "protein", "shellfish", 200),
    "lean beef (cooked)": _f(217, 26.1, 0.0, 11.8, 0.0, "protein", "meat", 200),
    "mutton (cooked)": _f(143, 27.1, 0.0, 3.0, 0.0, "protein", "meat", 200),
    "paneer": _f(265, 18.3, 1.2, 20.8, 0.0, "dairy", "dairy", 150),
    "tofu (firm)": _f(144, 17.3, 2.8, 8.7, 2.3, "protein", "soy", 250),
    "tempeh": _f(192, 20.3, 7.6, 10.8, 0.0, "protein", "soy", 200),
    "soya chunks (dry)": _f(345, 52.0, 33.0, 0.5, 13.0, "protein", "soy", 60),
    "whey protein powder": _f(400, 78.0, 8.0, 6.0, 0.0, "protein", "dairy", 60, ("scoop", 30, True)),
    # Dairy and alternatives
    "greek yogurt (low-fat)": _f(73, 10.0, 3.9, 1.9, 0.0, "dairy", "dairy", 300, ("cup", 245, False)),
    "curd (whole milk)": _f(61, 3.5, 4.7, 3.3, 0.0, "dairy", "dairy", 300, ("bowl", 150, False)),
    "milk (whole)": _f(61, 3.2, 4.8, 3.3, 0.0, "dairy", "dairy", 400, ("glass", 250, False)),
    "milk (low-fat)": _f(46, 3.4, 4.9, 1.5, 0.0, "dairy", "dairy", 400, ("glass", 250, False)),
    "milk (skimmed)": _f(34, 3.4, 5.0, 0.1, 0.0, "dairy", "dairy", 400, ("glass", 250, False)),
    "soy milk (unsweetened)": _f(33, 2.9, 1.7, 1.6, 0.4, "dairy", "soy", 400, ("glass", 250, False)),
    "cottage cheese (low-fat)": _f(84, 11.0, 4.3, 2.3, 0.0, "dairy", "dairy", 250),
    "cheddar cheese": _f(403, 24.9, 1.3, 33.1, 0.0, "dairy", "dairy", 40, ("slice", 20, True)),
    # Legumes
    "lentils / dal (cooked)": _f(116, 9.0, 20.1, 0.4, 7.9, "legume", "", 300, ("bowl", 150, False)),
    "chickpeas (cooked)": _f(164, 8.9, 27.4, 2.6, 7.6, "legume", "", 250, ("bowl", 150, False)),
    "kidney beans / rajma (cooked)": _f(127, 8.7, 22.8, 0.5, 6.4, "legume", "", 250, ("bowl", 150, False)),
    "black beans (cooked)": _f(132, 8.9, 23.7, 0.5, 8.7, "legume", "", 250, ("cup", 170, False)),
    "moong sprouts": _f(30, 3.0, 5.9, 0.2, 1.8, "legume", "", 200, ("bowl", 100, False)),
    "besan (chickpea flour)": _f(387, 22.4, 57.8, 6.7, 10.8, "legume", "", 80),
    "hummus": _f(166, 7.9, 14.3, 9.6, 6.0, "legume", "sesame", 100, ("tbsp", 15, False)),
    # Grains and starches
    "white rice (cooked)": _f(130, 2.7, 28.2, 0.3, 0.4, "grain", "", 300, ("cup", 160, False)),
    "brown rice (cooked)": _f(123, 2.7, 25.6, 1.0, 1.6, "grain", "", 300, ("cup", 160, False)),
    "roti / chapati": _f(270, 9.0, 50.0, 4.0, 7.0, "grain", "gluten", 200, ("roti", 40, True)),
    "oats (dry)": _f(389, 16.9, 66.3, 6.9, 10.6, "grain", "gluten", 100),
    "whole wheat bread": _f(247, 13.0, 41.3, 3.4, 6.8, "grain", "gluten", 120, ("slice", 30, True)),
    "white bread": _f(265, 9.0, 49.0, 3.2, 2.7, "grain", "gluten", 100, ("slice", 25, True)),
    "potato (boiled)": _f(87, 1.9, 20.1, 0.1, 1.8, "grain", "", 300),
    "sweet potato (baked)": _f(90, 2.0, 20.7, 0.2, 3.3, "grain", "", 300),
    "quinoa (cooked)": _f(120, 4.4, 21.3, 1.9, 2.8, "grain", "", 250, ("cup", 185, False)),
    "pasta (cooked)": _f(158, 5.8, 30.9, 0.9, 1.8, "grain", "gluten", 300, ("cup", 140, False)),
    "sweet corn (boiled)": _f(96, 3.4, 21.0, 1.5, 2.4, "grain", "", 200),
    "makhana (dry)": _f(347, 9.7, 76.9, 0.1, 14.5, "grain", "", 40),
    # Fruit
    "banana": _f(89, 1.1, 22.8, 0.3, 2.6, "fruit", "", 240, ("banana", 120, True)),
    "apple": _f(52, 0.3, 13.8, 0.2, 2.4, "fruit", "", 360, ("apple", 180, True)),
    "orange": _f(47, 0.9, 11.8, 0.1, 2.4, "fruit", "", 260, ("orange", 130, True)),
    "papaya": _f(43, 0.5, 10.8, 0.3, 1.7, "fruit", "", 300),
    "mango": _f(60, 0.8, 15.0, 0.4, 1.6, "fruit", "", 250),
    "strawberries": _f(32, 0.7, 7.7, 0.3, 2.0, "fruit", "", 250),
    "blueberries": _f(57, 0.7, 14.5, 0.3, 2.4, "fruit", "", 200),
    "watermelon": _f(30, 0.6, 7.6, 0.2, 0.4, "fruit", "", 400),
    "guava": _f(68, 2.6, 14.3, 1.0, 5.4, "fruit", "", 200, ("guava", 100, True)),
    "dates": _f(282, 2.5, 75.0, 0.4, 8.0, "fruit", "", 48, ("date", 8, True)),
    "raisins": _f(299, 3.1, 79.2, 0.5, 3.7, "fruit", "", 30),
    # Vegetables
    "broccoli": _f(34, 2.8, 6.6, 0.4, 2.6, "vegetable", "", 250),
    "spinach": _f(23, 2.9, 3.6, 0.4, 2.2, "vegetable", "", 200),
    "mixed vegetables (cooked)": _f(65, 2.9, 13.1, 0.2, 4.4, "vegetable", "", 300, ("bowl", 150, False)),
    "cucumber": _f(15, 0.7, 3.6, 0.1, 0.5, "vegetable", "", 300),
    "tomato": _f(18, 0.9, 3.9, 0.2, 1.2, "vegetable", "", 250),
    "onion": _f(40, 1.1, 9.3, 0.1, 1.7, "vegetable", "", 150),
    "carrot": _f(41, 0.9, 9.6, 0.2, 2.8, "vegetable", "", 200),
    "cauliflower": _f(25, 1.9, 5.0, 0.3, 2.0, "vegetable", "", 250),
    "green beans": _f(31, 1.8, 7.0, 0.2, 2.7, "vegetable", "", 250),
    "bell pepper": _f(31, 1.0, 6.0, 0.3, 2.1, "vegetable", "", 200),
    "mushrooms": _f(22, 3.1, 3.3, 0.3, 1.0, "vegetable", "", 200),
    "green peas (cooked)": _f(84, 5.4, 15.6, 0.2, 5.5, "vegetable", "", 200),
    # Nuts, seeds and fats
    "almonds": _f(579, 21.2, 21.6, 49.9, 12.5, "fat", "tree_nut", 40),
    "peanuts": _f(567, 25.8, 16.1, 49.2, 8.5, "fat", "peanut", 40),
    "peanut butter": _f(588, 25.0, 20.0, 50.0, 6.0, "fat", "peanut", 32, ("tbsp", 16, False)),
    "walnuts": _f(654, 15.2, 13.7, 65.2, 6.7, "fat", "tree_nut", 30),
    "cashews": _f(553, 18.2, 30.2, 43.9, 3.3, "fat", "tree_nut", 30),
    "chia seeds": _f(486, 16.5, 42.1, 30.7, 34.4, "fat", "", 25, ("tbsp", 12, False)),
    "flax seeds": _f(534, 18.3, 28.9, 42.2, 27.3, "fat", "", 20, ("tbsp", 10, False)),
    "pumpkin seeds": _f(559, 30.2, 10.7, 49.1, 6.0, "fat", "", 30),
    "cooking oil": _f(884, 0.0, 0.0, 100.0, 0.0, "fat", "", 15, ("tsp", 5, False)),
    "ghee": _f(876, 0.3, 0.0, 99.5, 0.0, "fat", "dairy", 15, ("tsp", 5, False)),
    "butter": _f(717, 0.9, 0.1, 81.1, 0.0, "fat", "dairy", 15, ("tsp", 5, False)),
    "avocado": _f(160, 2.0, 8.5, 14.7, 6.7, "fat", "", 150),
    "coconut (fresh)": _f(354, 3.3, 15.2, 33.5, 9.0, "fat", "", 50),
    "dark chocolate (70-85%)": _f(598, 7.8, 45.9, 42.6, 10.9, "sweet", "", 25),
    # Sweeteners
    "honey": _f(304, 0.3, 82.4, 0.0, 0.2, "sweet", "honey", 20, ("tsp", 7, False)),
    "sugar": _f(387, 0.0, 100.0, 0.0, 0.0, "sweet", "", 15, ("tsp", 4, False)),
    "jaggery": _f(383, 0.4, 98.0, 0.1, 0.0, "sweet", "", 20),
}

PROTEIN_FOOD_SHARE = 0.25  # at least this share of calories from protein
FAT_FOOD_SHARE = 0.6
STAPLE_GROUPS = {"grain", "fruit", "legume"}

# ---------------------------------------------------------------------------
# Diet and allergy filtering
# ---------------------------------------------------------------------------

DIET_EXCLUDES = {
    "non_vegetarian": set(),
    "pescatarian": {"meat"},
    "eggetarian": {"meat", "fish", "shellfish"},
    "vegetarian": {"meat", "fish", "shellfish", "egg"},
    "vegan": {"meat", "fish", "shellfish", "egg", "dairy", "honey"},
}

ALLERGY_TAG_PATTERNS = [
    (r"\bpeanut|\bgroundnut", {"peanut"}),
    (r"\bnuts?\b|\btree ?nut", {"tree_nut", "peanut"}),
    (r"\bdairy|\bmilk|\blactose|\bcasein|\bwhey", {"dairy"}),
    (r"\beggs?\b", {"egg"}),
    (r"\bsoy", {"soy"}),
    (r"\bgluten|\bwheat|\bco?eliac", {"gluten"}),
    (r"\bfish\b", {"fish"}),
    (r"\bseafood", {"fish", "shellfish"}),
    (r"\bshellfish|\bprawns?\b|\bshrimps?\b|\bcrabs?\b|\blobsters?\b", {"shellfish"}),
    (r"\bsesame|\btahini", {"sesame"}),
]

# Words that reveal an excluded food in free text (dish names, tips, swaps).
TAG_TERMS = {
    "meat": ["meat", "chicken", "mutton", "beef", "pork", "lamb", "turkey", "bacon", "ham", "sausage", "keema"],
    "fish": ["fish", "salmon", "tuna", "sardine", "mackerel", "cod", "tilapia"],
    "shellfish": ["prawn", "shrimp", "crab", "lobster"],
    "egg": ["egg", "omelette", "omelet"],
    "dairy": [
        r"(?<!soy )(?<!almond )(?<!coconut )(?<!oat )milk", "paneer", "curd", "yogurt", "yoghurt",
        "cheese", r"(?<!peanut )(?<!nut )butter", "ghee", "whey", "cream", "lassi", "raita", "dahi",
    ],
    "honey": ["honey"],
    "peanut": ["peanut", "groundnut"],
    "tree_nut": ["almond", "cashew", "walnut", "pistachio", "hazelnut", "pecan"],
    "soy": ["soy", "soya", "tofu", "tempeh", "edamame"],
    "gluten": ["wheat", "roti", "chapati", "bread", "pasta", "paratha", "naan", "oat", "barley", "semolina", "rava", "maida"],
    "sesame": ["sesame", "tahini", "hummus"],
}

_FILLER_WORDS = r"\b(allerg\w*|intoleran\w*|sensitiv\w*|avoid\w*|dislike\w*|hate|no|not|don'?t|do|eat|like|to|i|am|im|any|all|foods?)\b"
_NO_ALLERGY = {"none", "nil", "nothing", "n a", "na"}


def _allergy_phrases(text: str) -> list[str]:
    phrases = []
    for part in re.split(r"[,;/\n]|\band\b|\bor\b", text.lower()):
        part = re.sub(_FILLER_WORDS, " ", part)
        part = " ".join(re.sub(r"[^a-z ]", " ", part).split())
        if len(part) >= 3 and part not in _NO_ALLERGY:
            phrases.append(part)
    return phrases


def _phrase_matches(phrase: str, text: str) -> bool:
    variants = {phrase, phrase.removesuffix("s"), phrase.removesuffix("es")}
    for v in variants:
        if len(v) < 3:
            continue
        if len(v) >= 5 and v in text:
            return True
        if re.search(rf"\b{re.escape(v)}", text):
            return True
    return False


@dataclass(frozen=True)
class FoodRules:
    allowed: list[str]
    excluded_tags: frozenset[str]
    avoid_phrases: tuple[str, ...]

    def mentions_excluded(self, text: str) -> bool:
        text = text.lower()
        for tag in self.excluded_tags:
            for term in TAG_TERMS.get(tag, []):
                if re.search(rf"\b{term}(?:e?s)?\b", text):
                    return True
        return any(_phrase_matches(p, text) for p in self.avoid_phrases)


def food_rules(diet_type: DietType, allergies: str) -> FoodRules:
    """Work out which foods the user can eat from their diet type and allergy text."""
    tags = set(DIET_EXCLUDES[diet_type])
    lowered = allergies.lower()
    for pattern, allergy_tags in ALLERGY_TAG_PATTERNS:
        if re.search(pattern, lowered):
            tags |= allergy_tags
    phrases = tuple(_allergy_phrases(allergies))
    allowed = [
        name for name, food in FOODS.items()
        if not (food.tags & tags) and not any(_phrase_matches(p, name) for p in phrases)
    ]
    return FoodRules(allowed, frozenset(tags), phrases)


# ---------------------------------------------------------------------------
# Meal plans: totals and portion fitting
# ---------------------------------------------------------------------------


@dataclass
class Item:
    food: str
    grams: int


@dataclass
class Meal:
    name: str
    time: str
    dish: str
    items: list[Item]


MACROS = ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")


def item_macros(item: Item) -> dict[str, float]:
    f = FOODS[item.food]
    factor = item.grams / 100
    return {
        "kcal": f.kcal * factor,
        "protein_g": f.protein * factor,
        "carbs_g": f.carbs * factor,
        "fat_g": f.fat * factor,
        "fiber_g": f.fiber * factor,
    }


def totals(items: list[Item]) -> dict[str, float]:
    result = dict.fromkeys(MACROS, 0.0)
    for item in items:
        for key, value in item_macros(item).items():
            result[key] += value
    return result


def day_totals(meals: list[Meal]) -> dict[str, float]:
    return totals([item for meal in meals for item in meal.items])


def step_grams(name: str) -> int:
    food = FOODS[name]
    if food.unit and food.unit[2]:
        return food.unit[1]
    return 5 if food.max_g <= 60 else 10


def clamp_portion(name: str, grams: float) -> int:
    step = step_grams(name)
    rounded = round(grams / step) * step
    return int(min(max(rounded, step), max_step_multiple(name)))


def max_step_multiple(name: str) -> int:
    step = step_grams(name)
    return max(step, FOODS[name].max_g // step * step)


def normalize_meals(raw_meals: list[dict], allowed: list[str]) -> list[Meal]:
    """Build meals from model output: drop disallowed foods, merge duplicates, clamp portions."""
    allowed_set = set(allowed)
    meals = []
    for raw in raw_meals:
        grams_by_food: dict[str, float] = {}
        for raw_item in raw.get("items", []):
            name = raw_item.get("food")
            if name in allowed_set:
                grams_by_food[name] = grams_by_food.get(name, 0) + max(float(raw_item.get("grams", 0)), 0)
        items = [Item(name, clamp_portion(name, grams)) for name, grams in grams_by_food.items()]
        meals.append(Meal(
            name=str(raw.get("name", "")).strip() or f"Meal {len(meals) + 1}",
            time=str(raw.get("time", "")).strip(),
            dish=str(raw.get("dish", "")).strip(),
            items=items,
        ))
    return meals


def _change_one(meals: list[Meal], direction: int, eligible, prefer, meal_metric: str, highest_first: bool) -> bool:
    """Move one eligible item by one step, in the meal that needs it most."""
    ranked = sorted(meals, key=lambda m: totals(m.items)[meal_metric], reverse=highest_first)
    for meal in ranked:
        candidates = []
        for item in meal.items:
            food = FOODS[item.food]
            new_grams = item.grams + direction * step_grams(item.food)
            if eligible(food) and step_grams(item.food) <= new_grams <= max_step_multiple(item.food):
                candidates.append(item)
        if candidates:
            item = max(candidates, key=prefer)
            item.grams += direction * step_grams(item.food)
            return True
    return False


def fit_portions(meals: list[Meal], targets: dict, max_steps: int = 500) -> None:
    """Adjust portions in place until calories are within 5% and protein is met.

    Greedy, one step at a time: raise protein foods when protein is short, trim
    non-protein foods when calories run high, trim fats when fat runs high, and
    raise staples (grains, fruit, legumes) when calories run low. Every portion
    stays between one step and the food's sensible maximum.
    """
    kcal_t, protein_t, fat_t = targets["target_kcal"], targets["protein_g"], targets["fat_g"]

    def is_protein(f: Food) -> bool:
        return f.protein_share >= PROTEIN_FOOD_SHARE

    def kcal_per_step(i: Item) -> float:
        return FOODS[i.food].kcal * step_grams(i.food)

    def room(i: Item) -> float:
        return (max_step_multiple(i.food) - i.grams) / max_step_multiple(i.food)

    seen = set()
    for _ in range(max_steps):
        state = tuple(i.grams for m in meals for i in m.items)
        if state in seen:
            break
        seen.add(state)
        day = day_totals(meals)

        if day["protein_g"] < 0.95 * protein_t and _change_one(
            meals, +1, is_protein, lambda i: FOODS[i.food].protein_share, "protein_g", highest_first=False
        ):
            continue
        if day["kcal"] > 1.05 * kcal_t and (
            _change_one(meals, -1, lambda f: not is_protein(f), kcal_per_step, "kcal", highest_first=True)
            or (day["protein_g"] > 1.05 * protein_t
                and _change_one(meals, -1, is_protein, kcal_per_step, "kcal", highest_first=True))
        ):
            continue
        if day["fat_g"] > 1.25 * fat_t and _change_one(
            meals, -1, lambda f: f.fat_share >= FAT_FOOD_SHARE and not is_protein(f),
            lambda i: FOODS[i.food].fat * step_grams(i.food), "fat_g", highest_first=True,
        ):
            continue
        if day["kcal"] < 0.95 * kcal_t and (
            _change_one(meals, +1, lambda f: f.group in STAPLE_GROUPS, room, "kcal", highest_first=False)
            or _change_one(meals, +1, lambda f: f.fat_share < FAT_FOOD_SHARE, room, "kcal", highest_first=False)
        ):
            continue
        break


def household_measure(name: str, grams: int) -> str | None:
    """A kitchen-friendly amount, e.g. "2 rotis" or "about 1.5 cups"."""
    unit = FOODS[name].unit
    if unit is None:
        return None
    label, size, countable = unit
    count = grams / size
    if countable:
        count = round(count)
    else:
        count = round(count * 4) / 4 if count < 1 else round(count * 2) / 2
    if count <= 0:
        return None
    whole, part = divmod(count, 1)
    number = (f"{whole:g}" if whole else "") + {0: "", 0.25: "¼", 0.5: "½", 0.75: "¾"}[part]
    if label not in ("tsp", "tbsp") and count > 1:
        label = label + ("es" if label.endswith("s") else "s")
    return number + " " + label if countable else f"about {number} {label}"


def protein_foods(allowed: list[str], limit: int = 6) -> list[str]:
    ranked = sorted(allowed, key=lambda n: FOODS[n].protein_share, reverse=True)
    return [n for n in ranked if FOODS[n].protein_share >= PROTEIN_FOOD_SHARE][:limit]

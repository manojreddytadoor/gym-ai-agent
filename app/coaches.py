"""The Diet Coach and the Workout Coach.

Each coach asks the local model for a plan in a fixed JSON shape, then checks
and fixes it in code before showing it:

- the model can only pick foods and exercises from lists already filtered for
  the user's diet, allergies, equipment and injuries;
- every number shown (calories, macros, session length, weekly volume) is
  calculated here, and portions are fitted to the targets automatically;
- problems code can't fix are sent back to the model to revise.
"""

import json
import re
import threading
from collections.abc import Generator, Iterator
from dataclasses import dataclass

from .exercises import EXERCISES, allowed_exercises, format_amount, injury_tags
from .llm import LLMError, OllamaLLM
from .nutrition import (
    FOODS,
    NutritionTargetsInput,
    calculate_nutrition_targets,
    day_totals,
    fit_portions,
    food_rules,
    household_measure,
    item_macros,
    normalize_meals,
    protein_foods,
    totals,
)
from .profile import LABELS, UserProfile
from .training import MAJOR_MUSCLES, WorkoutPlanInput, analyze_workout_plan, session_minutes
from .training import Exercise as TimedExercise

Event = dict
MAX_REVISIONS = 2
HISTORY_TURNS = 3
WARMUP_MINUTES = 8
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Default training days when the model repeats a weekday.
SCHEDULES = {
    1: [0], 2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 4], 5: [0, 1, 2, 4, 5], 6: [0, 1, 2, 3, 4, 5], 7: list(range(7)),
}
VOLUME_GOALS = {"fat_loss", "muscle_gain", "recomposition", "strength"}


@dataclass
class Issue:
    problem: str  # shown to the user if it can't be fixed
    fix_hint: str  # extra instruction for the model

    def for_model(self) -> str:
        return f"{self.problem} {self.fix_hint}"


class Coach:
    """Shared flow: generate -> check and fix in code -> ask for revisions -> render."""

    name = "coach"

    def __init__(self, llm: OllamaLLM, profile: UserProfile):
        self.llm = llm
        self.profile = profile
        self.plan: dict | None = None
        self.history: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    # Subclasses provide these.
    def system_prompt(self) -> str: ...
    def create_prompt(self) -> str: ...
    def plan_context(self) -> str: ...
    def schema(self, follow_up: bool) -> dict: ...
    def check(self, raw: dict) -> tuple[dict, list[Issue], list[str]]: ...
    def render(self, plan: dict, notes: list[str]) -> str: ...
    def merge(self, old: dict, new: dict) -> dict:
        return new

    def chat(self, message: str | None = None) -> Iterator[Event]:
        try:
            if self.plan is None or not message:
                yield from self._create()
            else:
                yield from self._follow_up(message)
        except LLMError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        yield {"type": "done"}

    def _create(self) -> Iterator[Event]:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.create_prompt()},
        ]
        _, plan, notes = yield from self._generate_checked(messages, self.schema(follow_up=False))
        self.plan = plan
        self.history.clear()
        yield {"type": "reply", "markdown": self.render(plan, notes)}

    def _follow_up(self, message: str) -> Iterator[Event]:
        recent = "\n".join(f"User: {q}\nCoach: {a}" for q, a in self.history[-HISTORY_TURNS:])
        prompt = (
            f"{self.plan_context()}\n\n"
            + (f"Recent conversation:\n{recent}\n\n" if recent else "")
            + f"The user's new message: {message}\n\n"
            "Answer in `reply` in 2-4 plain sentences, without numbers the app calculates (calories, "
            "macros, minutes). If the message asks to change the plan, set update_plan to true and return "
            "the complete updated plan, not only the changed part. If it is only a question, set "
            "update_plan to false and leave the plan lists empty. If they ask for something that isn't in "
            "the allowed list, say so and use the closest allowed option."
        )
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        raw, plan, notes = yield from self._generate_checked(messages, self.schema(follow_up=True))
        reply = str(raw.get("reply", "")).strip()
        self.history.append((message, reply))
        if plan is None:
            yield {"type": "reply", "markdown": reply or "I don't have an answer to that."}
            return
        self.plan = self.merge(self.plan, plan)
        yield {"type": "reply", "markdown": f"{reply}\n\n{self.render(self.plan, notes)}"}

    def _generate_checked(self, messages: list[dict], schema: dict) -> Generator[Event, None, tuple]:
        status = "Writing the plan"
        for attempt in range(MAX_REVISIONS + 1):
            raw = yield from self.llm.generate_json(messages, schema, status=status, caller=self.name)
            if raw.get("update_plan") is False:
                return raw, None, []
            yield {"type": "status", "text": "Checking the plan"}
            plan, issues, notes = self.check(raw)
            if not issues:
                return raw, plan, notes
            if attempt == MAX_REVISIONS:
                return raw, plan, notes + [f"Couldn't fully fix: {i.problem}" for i in issues]
            status = f"Revising the plan to fix {len(issues)} problem{'s' * (len(issues) > 1)} the checks found"
            messages = messages + [
                {"role": "assistant", "content": json.dumps(raw)},
                {"role": "user", "content": (
                    "The app checked your plan and found these problems:\n- "
                    + "\n- ".join(i.for_model() for i in issues)
                    + "\nReturn the complete corrected plan in the same JSON format."
                )},
            ]


# ---------------------------------------------------------------------------
# Diet Coach
# ---------------------------------------------------------------------------

DIET_SYSTEM = """\
You are a sports-nutrition coach. You design a one-day meal plan for someone who trains at the gym. \
You choose the foods and rough portions; the app then calculates all calories and macros and fine-tunes \
the portions, so you don't need to do any arithmetic.

Rules:
- Only use foods from the allowed list, spelled exactly as listed. Build each meal from its ingredients, \
for example dal with rice = "lentils / dal (cooked)" + "white rice (cooked)" + "cooking oil".
- Every meal needs at least one protein-rich food. Hitting the protein target matters most.
- Give each meal a name (Breakfast, Lunch, Snack, Dinner, Pre-workout, Post-workout), a clock time, and a \
short dish name without quantities, such as "Paneer bhurji with roti".
- Put a carb-and-protein meal 1-3 hours before the workout and a protein-rich meal after it.
- Match the person's cuisine and budget, and make meals a real person would eat, with vegetables at lunch \
and dinner.
- Grams are for the food as eaten (cooked rice, dry oats).
- swaps: 3-5 short swaps, each replacing a food that is in your plan with an allowed food of a similar type.
- tips: 3-5 short eating-habit tips specific to this person, such as meal prep, timing or eating out. No \
numbers (the app shows calorie, protein and water targets), no exercise advice, no medical claims."""

GROUP_TITLES = {
    "protein": "Protein foods", "dairy": "Dairy and alternatives", "legume": "Legumes", "grain": "Grains and starches",
    "fruit": "Fruit", "vegetable": "Vegetables", "fat": "Nuts, seeds and fats", "sweet": "Sweeteners",
}


class DietCoach(Coach):
    name = "Diet Coach"

    def __init__(self, llm: OllamaLLM, profile: UserProfile):
        super().__init__(llm, profile)
        p = profile
        self.targets = calculate_nutrition_targets(NutritionTargetsInput(
            weight_kg=p.weight_kg, height_cm=p.height_cm, age=p.age, sex=p.sex, daily_activity=p.daily_activity,
            training_days_per_week=p.training_days_per_week, session_minutes=p.session_minutes, goal=p.goal,
        ))
        self.rules = food_rules(p.diet_type, p.allergies)

    def system_prompt(self) -> str:
        return DIET_SYSTEM

    def _foods_list(self) -> str:
        lines = []
        for group, title in GROUP_TITLES.items():
            names = [n for n in self.rules.allowed if FOODS[n].group == group]
            if names:
                lines.append(f"- {title}: " + "; ".join(f"{n} [{FOODS[n].protein:g} g protein]" for n in names))
        return "\n".join(lines)

    def _targets_line(self) -> str:
        t = self.targets
        return (f"Daily targets (calculated by the app): {t['target_kcal']} kcal, {t['protein_g']} g protein, "
                f"{t['carbs_g']} g carbs, {t['fat_g']} g fat.")

    def create_prompt(self) -> str:
        p = self.profile
        extra = ""
        if p.medical_conditions:
            extra = (f"\nThey reported these medical conditions: {p.medical_conditions}. Keep choices conservative "
                     "and make one tip say to confirm the plan with a doctor or registered dietitian.")
        return (
            f"{p.to_prompt()}\n\n{self._targets_line()}{extra}\n\n"
            f"Allowed foods (protein per 100 g in brackets):\n{self._foods_list()}\n\n"
            f"Create a one-day meal plan with exactly {p.meals_per_day} meals."
        )

    def plan_context(self) -> str:
        plan = {
            "meals": [
                {"name": m.name, "time": m.time, "dish": m.dish,
                 "items": [{"food": i.food, "grams": i.grams} for i in m.items]}
                for m in self.plan["meals"]
            ],
        }
        return (
            f"{self.profile.to_prompt()}\n\n{self._targets_line()}\n\n"
            f"Allowed foods (protein per 100 g in brackets):\n{self._foods_list()}\n\n"
            f"Current meal plan:\n{json.dumps(plan)}"
        )

    def schema(self, follow_up: bool) -> dict:
        item = {
            "type": "object",
            "properties": {
                "food": {"type": "string", "enum": self.rules.allowed},
                "grams": {"type": "integer", "minimum": 5, "maximum": 500},
            },
            "required": ["food", "grams"],
        }
        meal = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "time": {"type": "string"},
                "dish": {"type": "string"},
                "items": {"type": "array", "items": item, "minItems": 1, "maxItems": 8},
            },
            "required": ["name", "time", "dish", "items"],
        }
        meals = {"type": "array", "items": meal, "maxItems": 6}
        if not follow_up:
            meals["minItems"] = meals["maxItems"] = self.profile.meals_per_day
        strings = {"type": "array", "items": {"type": "string"}, "maxItems": 5}
        props = {"meals": meals, "swaps": strings, "tips": strings}
        required = ["meals", "swaps", "tips"]
        if follow_up:
            props = {"reply": {"type": "string"}, "update_plan": {"type": "boolean"}} | props
            required = ["reply", "update_plan", "meals"]
        return {"type": "object", "properties": props, "required": required}

    def check(self, raw: dict) -> tuple[dict, list[Issue], list[str]]:
        meals = normalize_meals(raw.get("meals", []), self.rules.allowed)
        for meal in meals:
            # Portions change during fitting, so counts like "2 rotis" in the name would go stale.
            meal.dish = " ".join(re.sub(r"\d+(\.\d+)?\s*(g|ml|x)?\b", "", meal.dish).split())
            if self.rules.mentions_excluded(meal.dish):
                meal.dish = ""  # a label like "egg bhurji" for a vegetarian would be misleading
        fit_portions(meals, self.targets)

        issues = []
        t = self.targets
        day = day_totals(meals)
        if len(meals) < 2:
            issues.append(Issue("The plan has fewer than 2 meals.", "Give the full day of meals."))
        for meal in meals:
            if not meal.items:
                issues.append(Issue(f"{meal.name} has no foods.", "Give every meal at least two foods."))
        if day["protein_g"] < 0.9 * t["protein_g"]:
            issues.append(Issue(
                f"Protein reaches only {day['protein_g']:.0f} g of the {t['protein_g']} g target, "
                "even with the largest sensible portions.",
                "Add protein-rich foods to more meals, such as " + ", ".join(protein_foods(self.rules.allowed)) + ".",
            ))
        if day["kcal"] < 0.9 * t["target_kcal"]:
            issues.append(Issue(
                f"Calories reach only {day['kcal']:.0f} of the {t['target_kcal']} kcal target, "
                "even with the largest sensible portions.",
                "Add more foods, such as grains, legumes or fruit.",
            ))
        elif day["kcal"] > 1.1 * t["target_kcal"]:
            issues.append(Issue(
                f"Calories are {day['kcal']:.0f} kcal, well over the {t['target_kcal']} kcal target, "
                "even with the smallest portions.",
                "Use fewer foods or lighter ones.",
            ))

        def clean(key: str) -> list[str]:
            return [s.strip() for s in raw.get(key, []) if s.strip() and not self.rules.mentions_excluded(s)]

        plan = {"meals": meals, "swaps": clean("swaps"), "tips": clean("tips")}
        return plan, issues, []

    def merge(self, old: dict, new: dict) -> dict:
        return {
            "meals": new["meals"],
            "swaps": new["swaps"] or old["swaps"],
            "tips": new["tips"] or old["tips"],
        }

    def render(self, plan: dict, notes: list[str]) -> str:
        t = self.targets
        p = self.profile
        lines = [
            "## Your daily targets", "",
            "| Calories | Protein | Carbs | Fat | Fiber | Water |",
            "|---|---|---|---|---|---|",
            f"| {t['target_kcal']} kcal | {t['protein_g']} g | {t['carbs_g']} g | {t['fat_g']} g | {t['fiber_g']} g "
            f"| {t['water_liters_rest_day']}-{t['water_liters_training_day']} L |",
            "",
        ]
        change = t["expected_weekly_weight_change_kg"]
        if abs(change) >= 0.05:
            direction = "loss" if change < 0 else "gain"
            lines.append(f"Your maintenance level is about {t['maintenance_kcal']} kcal a day, so expect roughly "
                         f"{abs(change):.2f} kg of weight {direction} per week.")
        else:
            lines.append(f"This is about your maintenance level ({t['maintenance_kcal']} kcal), so your weight "
                         "should stay stable.")
        lines += [f"- {note}" for note in t["notes"]]

        lines += ["", "## Meal plan"]
        for meal in plan["meals"]:
            m = totals(meal.items)
            lines += ["", f"### {meal.name}" + (f" · {meal.time}" if meal.time else "")]
            if meal.dish:
                lines.append(f"*{meal.dish}*")
            lines += ["", "| Food | Amount | kcal | Protein |", "|---|---|---|---|"]
            for item in meal.items:
                im = item_macros(item)
                measure = household_measure(item.food, item.grams)
                amount = f"{item.grams} g" + (f" ({measure})" if measure else "")
                lines.append(f"| {item.food[0].upper() + item.food[1:]} | {amount} | {im['kcal']:.0f} "
                             f"| {im['protein_g']:.0f} g |")
            lines += ["", f"**Meal total:** {m['kcal']:.0f} kcal · {m['protein_g']:.0f} g protein · "
                          f"{m['carbs_g']:.0f} g carbs · {m['fat_g']:.0f} g fat"]

        day = day_totals(plan["meals"])
        lines += [
            "", "## Day total", "",
            "| | Calories | Protein | Carbs | Fat | Fiber |", "|---|---|---|---|---|---|",
            f"| This plan | {day['kcal']:.0f} kcal | {day['protein_g']:.0f} g | {day['carbs_g']:.0f} g "
            f"| {day['fat_g']:.0f} g | {day['fiber_g']:.0f} g |",
            f"| Target | {t['target_kcal']} kcal | {t['protein_g']} g | {t['carbs_g']} g | {t['fat_g']} g "
            f"| {t['fiber_g']} g |",
            "",
        ]
        kcal_ok = abs(day["kcal"] - t["target_kcal"]) <= 0.05 * t["target_kcal"]
        protein_ok = day["protein_g"] >= 0.95 * t["protein_g"]
        if kcal_ok and protein_ok:
            lines.append("✅ Calories are within 5% of your target and your protein target is met.")
        else:
            lines.append("⚠️ This plan doesn't fully meet your "
                         + " and ".join(x for x, ok in (("calorie", kcal_ok), ("protein", protein_ok)) if not ok)
                         + " target. Ask the coach to adjust it.")

        if plan["swaps"]:
            lines += ["", "## Easy swaps"] + [f"- {s}" for s in plan["swaps"]]
        if plan["tips"]:
            lines += ["", "## Tips"] + [f"- {s}" for s in plan["tips"]]

        warnings = list(notes)
        if p.medical_conditions:
            warnings.append(f"You listed {p.medical_conditions}. Please check this plan with your doctor or a "
                            "registered dietitian before starting.")
        if p.age < 18:
            warnings.append("You're under 18, so involve a parent or doctor before changing your diet.")
        if warnings:
            lines += [""] + [f"> ⚠️ {w}" for w in warnings]
        diet = LABELS[p.diet_type].lower() + (f", avoiding {p.allergies}" if p.allergies else "")
        lines += ["", f"*All numbers are calculated by the app from its food database (USDA / IFCT values), not "
                      f"by the AI, and portions were fitted to your targets automatically. Foods were limited to "
                      f"ones that fit your diet: {diet}.*"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workout Coach
# ---------------------------------------------------------------------------


WORKOUT_SYSTEM = """\
You are a strength and conditioning coach designing one person's weekly gym program. You choose which \
exercises they do each day and how many sets. The app fills in reps, rest and effort from their goal, and \
checks session length and weekly volume, so you never do arithmetic.

Rules:
- Only use exercises from the allowed list, spelled exactly as listed. The list is already filtered for the \
person's equipment and injuries.
- Put the hardest compound exercises first and smaller isolation work last.
- Match the day's focus, and across the week train chest, back, shoulders, quads, hamstrings, glutes and \
arms at least twice when there are 3 or more training days.
- sets: 2-4 for most exercises, 5 only for a main lift."""

GOAL_GUIDE = {
    "fat_loss": "Keep heavy resistance training to hold on to muscle; short cardio finishers are fine.",
    "muscle_gain": "Favour compound lifts and enough volume per muscle.",
    "recomposition": "Favour compound lifts, with a little conditioning.",
    "strength": "Build the week around the main barbell or dumbbell lifts.",
    "general_fitness": "Mix strength work with some cardio or conditioning.",
    "endurance": "Moderate strength work plus a cardio finisher in most sessions.",
}

# The app prescribes these, so the model doesn't have to: (compound, isolation) rep ranges.
REP_RANGES = {
    "strength": ((4, 6), (8, 12)),
    "muscle_gain": ((6, 10), (10, 15)),
    "recomposition": ((6, 10), (10, 15)),
    "fat_loss": ((6, 10), (10, 15)),
    "general_fitness": ((8, 12), (12, 15)),
    "endurance": ((10, 15), (12, 20)),
}
# Proven day layouts, so a lopsided week (three chest days, one leg day) can't happen.
SPLIT_TEMPLATES: dict[int, list[tuple[str, list[str]]]] = {
    2: [("Full body", ["Full body A", "Full body B"])],
    3: [("Full body", ["Full body A", "Full body B", "Full body C"]),
        ("Push / Pull / Legs", ["Push", "Pull", "Legs"])],
    4: [("Upper / Lower", ["Upper body A", "Lower body A", "Upper body B", "Lower body B"]),
        ("Push / Pull / Legs / Upper", ["Push", "Pull", "Legs", "Upper body"])],
    5: [("Upper / Lower / Push / Pull / Legs", ["Upper body", "Lower body", "Push", "Pull", "Legs"]),
        ("Push / Pull / Legs / Upper / Lower", ["Push", "Pull", "Legs", "Upper body", "Lower body"])],
    6: [("Push / Pull / Legs twice", ["Push A", "Pull A", "Legs A", "Push B", "Pull B", "Legs B"]),
        ("Upper / Lower three times", ["Upper A", "Lower A", "Upper B", "Lower B", "Upper C", "Lower C"])],
    7: [("Push / Pull / Legs twice plus a light day",
         ["Push A", "Pull A", "Legs A", "Push B", "Pull B", "Legs B", "Light full body"])],
}
# What each kind of day has to cover, so an "Upper body" day can't skip back work.
FOCUS_MUSCLES = {
    "upper": ["chest", "back", "shoulders"],
    "lower": ["quads", "hamstrings", "glutes"],
    "legs": ["quads", "hamstrings", "glutes"],
    "push": ["chest", "shoulders"],
    "pull": ["back"],
    "full": ["chest", "back", "quads"],
}
TIMED_RANGE = (30, 45)  # seconds, for planks and similar
CARDIO_RANGE = (10, 15)  # minutes
RIR_BY_EXPERIENCE = {"beginner": 3, "intermediate": 2, "advanced": 1}
MINUTES_PER_EXERCISE = 6.75  # 3 sets of work plus rest and setup
MAX_VOLUME_FIXES = 2


@dataclass
class PlannedExercise:
    name: str
    sets: int
    reps_low: int
    reps_high: int
    rest_seconds: int
    rir: int

    def timed(self) -> TimedExercise:
        ex = EXERCISES[self.name]
        reps = {"reps": f"{self.reps_low}-{self.reps_high}", "seconds": f"{self.reps_high}s",
                "minutes": f"{self.reps_high} min"}[ex.unit]
        return TimedExercise(
            name=self.name, primary_muscle=ex.primary, secondary_muscles=list(ex.secondary),
            sets=self.sets, reps=reps, rest_seconds=self.rest_seconds,
        )


@dataclass
class Day:
    weekday: str
    focus: str
    warmup: str
    exercises: list[PlannedExercise]


def _clamp(value, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return min(max(number, low), high)


def prescribe(name: str, sets: int, goal: str, experience: str) -> PlannedExercise:
    """Fill in reps, rest and effort for an exercise the model picked."""
    ex = EXERCISES[name]
    if ex.unit == "seconds":
        low, high = TIMED_RANGE
    elif ex.unit == "minutes":
        low, high = CARDIO_RANGE
    else:
        low, high = REP_RANGES[goal][0 if ex.compound else 1]

    if ex.primary == "cardio":
        rest = 0
    elif ex.unit != "reps":
        rest = 45
    elif ex.compound:
        rest = 180 if goal == "strength" else 120
    else:
        rest = 75

    rir = RIR_BY_EXPERIENCE[experience]
    if goal == "strength" and ex.compound:
        rir = min(rir + 1, 4)
    return PlannedExercise(name, 1 if ex.unit == "minutes" else sets, low, high, rest, rir)


class WorkoutCoach(Coach):
    """Builds the week one day at a time: short model calls, visible progress."""

    name = "Workout Coach"

    def __init__(self, llm: OllamaLLM, profile: UserProfile):
        super().__init__(llm, profile)
        self.allowed = allowed_exercises(profile.equipment, profile.injuries)
        self.allowed_set = set(self.allowed)
        self.avoided_joints = sorted(injury_tags(profile.injuries))
        self.trainable = {EXERCISES[n].primary for n in self.allowed}
        self.trainable_without_injuries = {EXERCISES[n].primary for n in allowed_exercises(profile.equipment, "")}
        self.max_exercises = _clamp((profile.session_minutes - WARMUP_MINUTES) / MINUTES_PER_EXERCISE, 3, 7)
        self.min_exercises = min(4, self.max_exercises)

    # ---- prompts -----------------------------------------------------------

    def _exercise_list(self) -> str:
        by_muscle: dict[str, list[str]] = {}
        for name in self.allowed:
            ex = EXERCISES[name]
            label = name + (f" [{ex.unit}]" if ex.unit != "reps" else "")
            by_muscle.setdefault(ex.primary, []).append(label)
        return "\n".join(f"- {muscle}: " + "; ".join(names) for muscle, names in by_muscle.items())

    def system_prompt(self) -> str:
        """Stable across every call in a session, so Ollama can reuse the cached prompt."""
        p = self.profile
        return (
            f"{WORKOUT_SYSTEM}\n\n{p.to_prompt()}\n\n"
            f"Goal guidance: {GOAL_GUIDE[p.goal]}\n"
            f"Sessions are {p.session_minutes} minutes including a warm-up, which fits about "
            f"{self.max_exercises} exercises.\n\n"
            f"Allowed exercises, by main muscle:\n{self._exercise_list()}"
        )

    def _day_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "exercises": {
                    "type": "array",
                    "minItems": self.min_exercises,
                    "maxItems": self.max_exercises,
                    "items": {
                        "type": "object",
                        "properties": {
                            "exercise": {"type": "string", "enum": self.allowed},
                            "sets": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["exercise", "sets"],
                    },
                },
            },
            "required": ["exercises"],
        }

    # ---- building the week -------------------------------------------------

    def _create(self) -> Iterator[Event]:
        p = self.profile
        count = p.training_days_per_week
        weekdays = [WEEKDAYS[i] for i in SCHEDULES[count]]

        templates = SPLIT_TEMPLATES[count]
        chosen = templates[0]
        if len(templates) > 1:
            answer = yield from self.llm.generate_json(
                [{"role": "system", "content": self.system_prompt()},
                 {"role": "user", "content": "Which split suits this person best for "
                                             f"{count} training days a week?"}],
                {"type": "object", "properties": {"split": {"type": "string", "enum": [t[0] for t in templates]}},
                 "required": ["split"]},
                status="Choosing your training split", caller=self.name,
            )
            chosen = next((t for t in templates if t[0] == answer.get("split")), templates[0])
        split_name, focuses = chosen

        days: list[Day] = []
        for number, (weekday, focus) in enumerate(zip(weekdays, focuses), start=1):
            day = yield from self._ask_day(weekday, focus, days, f"day {number} of {count}: {focus}")
            days.append(day)

        self._trim_overshoot(days)
        days, notes = yield from self._balance_volume(days)
        self.plan = {"split": split_name, "days": days}
        self.history.clear()
        yield {"type": "reply", "markdown": self.render(self.plan, notes)}

    def required_muscles(self, focus: str) -> list[str]:
        """The muscle groups a day with this focus has to train."""
        key = focus.split()[0].lower() if focus.split() else ""
        return [m for m in FOCUS_MUSCLES.get(key, []) if m in self.trainable]

    def _day_messages(self, weekday: str, focus: str, done: list[Day], extra: str = "") -> list[dict]:
        used = ", ".join(sorted({e.name for day in done for e in day.exercises}))
        required = self.required_muscles(focus)
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": (
                f"{weekday} - focus: {focus}.\n"
                + (f"Already used on other days: {used}.\n" if used else "")
                + f"Choose {self.min_exercises}-{self.max_exercises} exercises for this day, compound lifts "
                  "first, and the number of sets for each."
                + (f"\nInclude at least one exercise for each of: {', '.join(required)}." if required else "")
                + (f"\n{extra}" if extra else "")
            )},
        ]

    def _ask_day(self, weekday: str, focus: str, done: list[Day], label: str
                 ) -> Generator[Event, None, Day]:
        """Build one day, re-asking once if it misses a muscle group its focus needs."""
        raw = yield from self.llm.generate_json(
            self._day_messages(weekday, focus, done), self._day_schema(),
            status=f"Building {label}", caller=self.name,
        )
        day = self._build_day(weekday, focus, raw.get("exercises", []))
        missing = self._missing_muscles(day, focus)
        if missing:
            raw = yield from self.llm.generate_json(
                self._day_messages(weekday, focus, done,
                                   extra=f"Your last attempt had no {', '.join(missing)} exercise. This day "
                                         f"must include one for each of: {', '.join(missing)}."),
                self._day_schema(),
                status=f"Adding {', '.join(missing)} to {weekday}", caller=self.name,
            )
            day = self._build_day(weekday, focus, raw.get("exercises", []))
            for muscle in self._missing_muscles(day, focus):
                self._insert_exercise(day, muscle)  # the model wouldn't; do it in code
        return day

    def _missing_muscles(self, day: Day, focus: str) -> list[str]:
        trained = {EXERCISES[e.name].primary for e in day.exercises}
        return [m for m in self.required_muscles(focus) if m not in trained]

    def _insert_exercise(self, day: Day, muscle: str) -> None:
        used = {e.name for e in day.exercises}
        options = [n for n in self.allowed if EXERCISES[n].primary == muscle and n not in used]
        if not options:
            return
        pick = min(options, key=lambda n: (not EXERCISES[n].compound, n))
        if len(day.exercises) >= self.max_exercises:
            spare = next((e for e in reversed(day.exercises) if not EXERCISES[e.name].compound), None)
            day.exercises.remove(spare or day.exercises[-1])
        day.exercises.append(prescribe(pick, 3, self.profile.goal, self.profile.experience))
        day.exercises.sort(key=lambda e: (EXERCISES[e.name].primary == "cardio", not EXERCISES[e.name].compound))
        self._fit_to_time(day)

    def _build_day(self, weekday: str, focus: str, raw_exercises: list[dict]) -> Day:
        seen: set[str] = set()
        exercises = []
        for raw in raw_exercises:
            name = raw.get("exercise")
            if name not in self.allowed_set or name in seen:
                continue
            seen.add(name)
            exercises.append(prescribe(name, _clamp(raw.get("sets"), 1, 5), self.profile.goal, self.profile.experience))
        # Compounds first, conditioning last; the model's order is kept within each group.
        exercises.sort(key=lambda e: (EXERCISES[e.name].primary == "cardio", not EXERCISES[e.name].compound))
        first = exercises[0].name.lower() if exercises else "the first exercise"
        day = Day(weekday, focus or "Training", f"5 minutes of easy cardio and mobility, then 2 light "
                                                f"ramp-up sets of {first}.", exercises)
        self._fit_to_time(day)
        return day

    def _fit_to_time(self, day: Day) -> None:
        """Trim sets from the end of the session until it fits the time available."""
        limit = self.profile.session_minutes + 3
        while day.exercises and session_minutes([e.timed() for e in day.exercises], WARMUP_MINUTES) > limit:
            trimmable = [e for e in reversed(day.exercises) if e.sets > 2]
            if trimmable:
                trimmable[0].sets -= 1
            elif len(day.exercises) > 3:
                day.exercises.pop()
            else:
                break

    def _analysis(self, days: list[Day]) -> dict:
        days = [d for d in days if d.exercises]
        if not days:
            return {"weekly_volume": {}, "sessions": []}
        return analyze_workout_plan(WorkoutPlanInput(
            days=[{"day": d.weekday, "exercises": [e.timed() for e in d.exercises]} for d in days],
            experience=self.profile.experience,
            session_minutes_available=self.profile.session_minutes,
            warmup_minutes=WARMUP_MINUTES,
        ))

    def _short_muscles(self, days: list[Day]) -> list[str]:
        """Major muscles below their weekly range that this person could still train."""
        if self.profile.goal not in VOLUME_GOALS:
            return []
        volume = self._analysis(days)["weekly_volume"]
        return [m for m in MAJOR_MUSCLES
                if m in self.trainable and (volume.get(m) or {}).get("status") == "below range"]

    def _trim_overshoot(self, days: list[Day], max_steps: int = 150) -> None:
        """Drop sets where a muscle gets more weekly work than its range allows.

        Five training days can otherwise pile 20+ chest sets on a beginner. Sets
        come off isolation work late in the week first, so the main lifts stay.
        """
        # A muscle worked only as a helper (triceps under pressing) can stay over
        # range with nothing left to trim; park it and keep fixing the others.
        stuck: set[str] = set()
        for _ in range(max_steps):
            volume = self._analysis(days)["weekly_volume"]
            over = [m for m in MAJOR_MUSCLES
                    if m not in stuck and (volume.get(m) or {}).get("status") == "above range"]
            if not over:
                return
            muscle = max(over, key=lambda m: volume[m]["weekly_sets"])
            # Only consider exercises that can still lose a set or be dropped, or one
            # maxed-out day would block every other day from being trimmed.
            candidates = [(index, e) for index, day in enumerate(days) for e in day.exercises
                          if EXERCISES[e.name].primary == muscle and (e.sets > 2 or len(day.exercises) > 3)]
            if not candidates:
                stuck.add(muscle)
                continue
            # Take sets off before dropping an exercise, isolation before compounds, later days first.
            index, target = max(candidates, key=lambda c: (c[1].sets > 2, not EXERCISES[c[1].name].compound,
                                                           c[1].sets, c[0]))
            if target.sets > 2:
                target.sets -= 1
            else:
                days[index].exercises.remove(target)

    def _balance_volume(self, days: list[Day]) -> Generator[Event, None, tuple[list[Day], list[str]]]:
        """Rebuild the emptiest day until every major muscle gets enough weekly sets."""
        rebuilt: set[str] = set()
        for _ in range(MAX_VOLUME_FIXES):
            short = self._short_muscles(days)
            candidates = [d for d in days if d.weekday not in rebuilt]
            if not short or not candidates:
                break
            # Rebuild the emptiest day, and never the same day twice in one pass.
            target = min(candidates, key=lambda d: session_minutes([e.timed() for e in d.exercises], WARMUP_MINUTES))
            rebuilt.add(target.weekday)
            wanted = ", ".join(short[:3])
            raw = yield from self.llm.generate_json(
                self._day_messages(target.weekday, target.focus, [d for d in days if d is not target],
                                   extra=f"This week is short on {wanted}. Include at least one exercise for "
                                         f"each of those, keeping the day's focus."),
                self._day_schema(),
                status=f"Adding {wanted} work to {target.weekday}", caller=self.name,
            )
            days[days.index(target)] = self._build_day(target.weekday, target.focus, raw.get("exercises", []))

        notes = []
        for muscle in MAJOR_MUSCLES:
            if muscle in self.trainable:
                continue
            if muscle in self.trainable_without_injuries:
                notes.append(f"Every {muscle} exercise in the list loads a joint you mentioned, so {muscle} gets "
                             "no direct work. A physiotherapist can suggest safe options.")
            else:
                notes.append(f"Your equipment has no direct {muscle} exercise, so {muscle} only gets indirect "
                             "work from other exercises. Adding dumbbells or resistance bands would fix this.")
        for muscle in self._short_muscles(days):
            notes.append(f"{muscle.capitalize()} is still below the weekly set range; ask for more {muscle} work "
                         "if you want it.")
        return days, notes

    # ---- follow-up questions ----------------------------------------------

    def _follow_up(self, message: str) -> Iterator[Event]:
        weekdays = [d.weekday for d in self.plan["days"]]
        current = {d.weekday: [f"{e.name} {e.sets} sets" for e in d.exercises] for d in self.plan["days"]}
        recent = "\n".join(f"User: {q}\nCoach: {a}" for q, a in self.history[-HISTORY_TURNS:])
        schema = {
            "type": "object",
            "properties": {
                "reply": {"type": "string", "maxLength": 600},
                "change_day": {"type": "string", "enum": ["none", *weekdays]},
                "exercises": self._day_schema()["properties"]["exercises"] | {"minItems": 0},
            },
            "required": ["reply", "change_day", "exercises"],
        }
        prompt = (
            f"Current program:\n{json.dumps(current)}\n\n"
            + (f"Recent conversation:\n{recent}\n\n" if recent else "")
            + f"The user's new message: {message}\n\n"
            "Answer in `reply` in 2-4 plain sentences, without numbers the app calculates (minutes, weekly "
            "sets). If they want one day changed, set change_day to that weekday and give that day's full "
            "exercise list; otherwise set change_day to \"none\" and leave exercises empty. If they ask for an "
            "exercise that isn't in the allowed list, say so and use the closest allowed one."
        )
        raw = yield from self.llm.generate_json(
            [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": prompt}],
            schema, status="Answering your question", caller=self.name,
        )
        reply = str(raw.get("reply", "")).strip()
        self.history.append((message, reply))

        changed = raw.get("change_day", "none")
        exercises = raw.get("exercises") or []
        if changed not in weekdays or not exercises:
            yield {"type": "reply", "markdown": reply or "I don't have an answer to that."}
            return
        index = weekdays.index(changed)
        old = self.plan["days"][index]
        self.plan["days"][index] = self._build_day(old.weekday, old.focus, exercises)
        self._trim_overshoot(self.plan["days"])
        _, notes = yield from self._balance_volume(self.plan["days"])
        yield {"type": "reply", "markdown": f"{reply}\n\n{self.render(self.plan, notes)}"}

    # ---- output ------------------------------------------------------------

    def _progression(self) -> list[str]:
        p = self.profile
        points = [
            "Log every session. Each week try to add a rep or two to each exercise while keeping your form.",
            "When you hit the top of the rep range on all sets of an exercise, add the smallest weight step "
            "and start again at the bottom of the range.",
        ]
        if p.goal == "strength":
            points.append("On the main lifts, add weight only when all sets feel controlled with 2 reps left in "
                          "the tank.")
        points.append("Every 5-6 weeks take an easier week: same exercises, about half the sets.")
        return points

    def _recovery(self) -> list[str]:
        p = self.profile
        steps = "8,000-10,000" if p.goal in ("fat_loss", "recomposition") else "7,000-8,000"
        points = [
            f"On rest days stay lightly active: aim for about {steps} steps.",
            "Sleep 7-9 hours. Muscle is rebuilt while you sleep, not while you train.",
            "Leave a day between hard sessions for the same muscles, and expect some soreness in the first weeks.",
        ]
        if p.goal in ("fat_loss", "general_fitness", "endurance"):
            points.append("Add 10-20 minutes of easy cardio after 2-3 sessions a week if you have the time.")
        return points

    def render(self, plan: dict, notes: list[str]) -> str:
        p = self.profile
        analysis = self._analysis(plan["days"])
        minutes = {s["day"]: s["estimated_minutes"] for s in analysis["sessions"]}
        schedule = ", ".join(d.weekday[:3] for d in plan["days"])
        lines = [
            "## Program overview", "",
            f"**{plan['split']}** · {len(plan['days'])} days a week ({schedule})", "",
            f"Sessions are built to fit your {p.session_minutes} minutes, with reps, rest and effort set for "
            f"{LABELS[p.goal].lower()} at your experience level.",
        ]
        short = [s["day"] for s in analysis["sessions"] if s["status"].startswith("well under")]
        if short:
            lines += ["", f"{', '.join(short)} finish well inside your {p.session_minutes} minutes: your weekly "
                          "sets are already at target. Add a set to the first two exercises if you want longer "
                          "sessions."]

        for day in plan["days"]:
            title = f"### {day.weekday}" + (f": {day.focus}" if day.focus else "")
            if day.weekday in minutes:
                title += f" · about {minutes[day.weekday]} min"
            lines += ["", title, "", f"**Warm-up:** {day.warmup}", "",
                      "| Exercise | Sets × Reps | Rest | RIR | Key cue |", "|---|---|---|---|---|"]
            for e in day.exercises:
                ex = EXERCISES[e.name]
                rest = "–" if e.rest_seconds == 0 else (
                    f"{e.rest_seconds} s" if e.rest_seconds < 120 else f"{e.rest_seconds / 60:g} min")
                rir = "–" if ex.primary == "cardio" or ex.unit != "reps" else str(e.rir)
                lines.append(f"| {e.name} | {e.sets} × {format_amount(e.name, e.reps_low, e.reps_high)} "
                             f"| {rest} | {rir} | {ex.cue} |")

        lines += ["", "## Weekly volume check", "", "| Muscle | Sets per week | Target | |", "|---|---|---|---|"]
        for muscle in MAJOR_MUSCLES:
            entry = analysis["weekly_volume"].get(muscle)
            if entry:
                mark = {"in range": "✅", "below range": "⬇️ low", "above range": "⬆️ high"}[entry["status"]]
                lines.append(f"| {muscle.capitalize()} | {entry['weekly_sets']:g} | {entry['target_range']} | {mark} |")
        lines += ["", "*Sets for a muscle's main exercises count fully; exercises that work it as a helper count half.*"]

        lines += ["", "## Progression"] + [f"- {s}" for s in self._progression()]
        lines += ["", "## Recovery and cardio"] + [f"- {s}" for s in self._recovery()]

        warnings = list(notes)
        if self.avoided_joints:
            joints = ", ".join(j.replace("_", " ") for j in self.avoided_joints)
            warnings.append(f"Because you mentioned {p.injuries}, exercises that load your {joints} were left out. "
                            "Stop any exercise that causes pain, and see a physiotherapist if it persists.")
        if p.medical_conditions:
            warnings.append(f"You listed {p.medical_conditions}. Get clearance from your doctor before starting.")
        if p.age < 18 or p.experience == "beginner":
            warnings.append("Focus on good technique with moderate weights; if you can, have a coach check your "
                            "form on the main lifts.")
        if warnings:
            lines += [""] + [f"> ⚠️ {w}" for w in warnings]
        lines += ["", f"*The AI chose the exercises and sets from the app's list, filtered for your equipment "
                      f"({LABELS[p.equipment].lower()}) and injuries. Reps, rest, effort, session times and the "
                      f"weekly volume check are calculated by the app.*"]
        return "\n".join(lines)

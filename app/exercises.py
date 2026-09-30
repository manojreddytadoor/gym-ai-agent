"""Exercise database for the Workout Coach, filtered by equipment and injuries.

The language model may only choose exercises from this list, so every exercise
in a plan is one the user can do with their equipment, avoids joints they've
reported as injured, and has known muscle targets for the volume check.
"""

import re
from dataclasses import dataclass

from .profile import Equipment

EQUIPMENT_AVAILABLE: dict[str, set[str]] = {
    "full_gym": {"barbell", "dumbbell", "cable", "machine", "bench", "pullup_bar", "bodyweight"},
    "home_gym": {"barbell", "dumbbell", "bench", "pullup_bar", "bodyweight"},
    "dumbbells_only": {"dumbbell", "bench", "bodyweight"},
    "bodyweight_only": {"bodyweight"},
}

INJURY_PATTERNS = [
    (r"\bknees?\b|\bacl\b|\bmeniscus|\bpatell", "knee"),
    (r"\bback\b|\bspin(e|al)\b|\bdiscs?\b|\bsciatica|\blumbar", "lower_back"),
    (r"\bshoulders?\b|\brotator|\blabrum|\bimpingement", "shoulder"),
    (r"\bwrists?\b|\bcarpal", "wrist"),
    (r"\belbows?\b|\bgolfer", "elbow"),
    (r"\bhips?\b", "hip"),
    (r"\bankles?\b", "ankle"),
]


@dataclass(frozen=True)
class Exercise:
    equipment: frozenset[str]
    primary: str
    secondary: tuple[str, ...]
    stresses: frozenset[str]  # joints this exercise loads heavily
    compound: bool
    unit: str  # "reps", "seconds" or "minutes"
    cue: str


def _e(equipment, primary, secondary="", stresses="", compound=False, unit="reps", cue="") -> Exercise:
    return Exercise(
        frozenset(equipment.split()), primary, tuple(secondary.split()), frozenset(stresses.split()),
        compound, unit, cue,
    )


EXERCISES: dict[str, Exercise] = {
    # Chest
    "Barbell bench press": _e("barbell bench", "chest", "triceps shoulders", "shoulder wrist", True,
        cue="Shoulder blades squeezed, lower the bar to mid-chest, press up and slightly back."),
    "Incline barbell bench press": _e("barbell bench", "chest", "shoulders triceps", "shoulder", True,
        cue="Bench at 30-45 degrees, bar to upper chest, elbows about 45 degrees from your body."),
    "Dumbbell bench press": _e("dumbbell bench", "chest", "triceps shoulders", "shoulder", True,
        cue="Lower the dumbbells to chest level with control, press up without clanking them."),
    "Incline dumbbell press": _e("dumbbell bench", "chest", "shoulders triceps", "shoulder", True,
        cue="Bench at 30-45 degrees, press up and slightly in, shoulder blades pulled back."),
    "Machine chest press": _e("machine", "chest", "triceps shoulders", "", True,
        cue="Handles at mid-chest height, press out smoothly, don't let shoulders roll forward."),
    "Dumbbell fly": _e("dumbbell bench", "chest", "", "shoulder",
        cue="Slight bend in the elbows, open wide until you feel a chest stretch, hug back up."),
    "Cable fly": _e("cable", "chest",
        cue="Step forward, slight elbow bend, bring hands together in an arc at chest height."),
    "Push-up": _e("bodyweight", "chest", "triceps shoulders core", "wrist", True,
        cue="Body in a straight line, chest to just above the floor, elbows about 45 degrees."),
    "Incline push-up (hands on a bench or table)": _e("bodyweight", "chest", "triceps shoulders", "wrist", True,
        cue="Easier push-up: the higher your hands, the easier it gets."),
    "Decline push-up (feet raised)": _e("bodyweight", "chest", "shoulders triceps", "wrist shoulder", True,
        cue="Feet on a chair or step, hips in line, lower your chest toward the floor."),
    "Parallel bar dip": _e("machine", "chest", "triceps shoulders", "shoulder elbow", True,
        cue="Lean slightly forward, lower until upper arms are parallel to the floor, no deeper."),
    # Back
    "Pull-up": _e("pullup_bar", "back", "biceps forearms", "shoulder elbow", True,
        cue="Start from a dead hang, pull your chest toward the bar, lower fully with control."),
    "Chin-up": _e("pullup_bar", "back", "biceps", "shoulder elbow", True,
        cue="Palms facing you, pull until your chin clears the bar, lower slowly."),
    "Lat pulldown": _e("cable", "back", "biceps", "", True,
        cue="Pull the bar to your upper chest, elbows down and back, only a slight lean."),
    "Seated cable row": _e("cable", "back", "biceps", "", True,
        cue="Chest tall, pull the handle to your stomach, squeeze shoulder blades together."),
    "Machine row": _e("machine", "back", "biceps", "", True,
        cue="Chest against the pad, drive elbows back, pause and squeeze."),
    "Barbell row": _e("barbell", "back", "biceps hamstrings", "lower_back", True,
        cue="Hinge to about 45 degrees with a flat back, pull the bar to your lower ribs."),
    "One-arm dumbbell row": _e("dumbbell bench", "back", "biceps", "", True,
        cue="Hand and knee on the bench, flat back, pull the dumbbell toward your hip."),
    "Chest-supported dumbbell row": _e("dumbbell bench", "back", "biceps shoulders", "", True,
        cue="Lie chest-down on an incline bench, row both dumbbells, squeeze shoulder blades."),
    "Inverted row (under a sturdy table or low bar)": _e("bodyweight", "back", "biceps core", "", True,
        cue="Body straight, pull your chest to the edge; bend your knees to make it easier."),
    "Prone Y-T-W raise": _e("bodyweight", "shoulders", "back",
        cue="Lie face down, lift your arms into Y, T and W shapes with thumbs up, hold 2 seconds each."),
    "Face pull": _e("cable", "shoulders", "back",
        cue="Rope at face height, pull toward your forehead with elbows high, pull the rope apart."),
    # Hips and hamstrings
    "Barbell deadlift": _e("barbell", "glutes", "hamstrings back forearms", "lower_back", True,
        cue="Bar over mid-foot, flat back, push the floor away and stand tall. Never round your back."),
    "Romanian deadlift": _e("barbell", "hamstrings", "glutes back", "lower_back", True,
        cue="Soft knees, push hips back until you feel a hamstring stretch, keep the bar close."),
    "Dumbbell Romanian deadlift": _e("dumbbell", "hamstrings", "glutes back", "lower_back", True,
        cue="Soft knees, hips back, dumbbells slide down your thighs, flat back throughout."),
    "Single-leg Romanian deadlift": _e("bodyweight", "hamstrings", "glutes core", "", True,
        cue="Hinge on one leg with the other reaching back, hips level; hold a support if needed."),
    "Barbell hip thrust": _e("barbell bench", "glutes", "hamstrings", "", True,
        cue="Upper back on the bench, drive through your heels, squeeze glutes at the top, chin tucked."),
    "Dumbbell hip thrust": _e("dumbbell bench", "glutes", "hamstrings", "", True,
        cue="Dumbbell on your hips, upper back on the bench, pause and squeeze at the top."),
    "Glute bridge": _e("bodyweight", "glutes", "hamstrings",
        cue="On your back, feet flat, drive hips up and squeeze your glutes for 2 seconds."),
    "Single-leg glute bridge": _e("bodyweight", "glutes", "hamstrings",
        cue="One foot planted, other leg up, lift your hips level without twisting."),
    "Cable pull-through": _e("cable", "glutes", "hamstrings", "", True,
        cue="Face away from the low cable, hinge back, then drive your hips forward to stand tall."),
    "Back extension": _e("machine", "glutes", "hamstrings back", "lower_back",
        cue="Hinge at the hips, rise until your body is straight, don't over-arch."),
    "Lying leg curl": _e("machine", "hamstrings",
        cue="Hips pressed into the pad, curl heels toward your glutes, lower slowly."),
    "Seated leg curl": _e("machine", "hamstrings",
        cue="Thigh pad snug, curl down fully, control the way back."),
    "Nordic hamstring curl (feet anchored)": _e("bodyweight", "hamstrings", "", "knee",
        cue="Kneel with ankles anchored, lower forward as slowly as you can, push back up with your hands."),
    # Shoulders
    "Overhead barbell press": _e("barbell", "shoulders", "triceps core", "shoulder lower_back", True,
        cue="Squeeze glutes and brace, press the bar straight up, head through at the top."),
    "Seated dumbbell shoulder press": _e("dumbbell bench", "shoulders", "triceps", "shoulder", True,
        cue="Back against the bench, press up, lower the dumbbells to ear level."),
    "Machine shoulder press": _e("machine", "shoulders", "triceps", "shoulder", True,
        cue="Handles at shoulder height, press up smoothly, back on the pad."),
    "Dumbbell lateral raise": _e("dumbbell", "shoulders",
        cue="Slight elbow bend, raise to shoulder height leading with your elbows, no swinging."),
    "Cable lateral raise": _e("cable", "shoulders",
        cue="Cable from the low pulley across your body, raise to shoulder height slowly."),
    "Reverse dumbbell fly": _e("dumbbell", "shoulders", "back",
        cue="Hinge forward, arms slightly bent, raise out to the sides squeezing your rear shoulders."),
    "Pike push-up": _e("bodyweight", "shoulders", "triceps", "shoulder wrist", True,
        cue="Hips high in an upside-down V, lower your head toward the floor between your hands."),
    # Arms
    "Barbell curl": _e("barbell", "biceps", "forearms", "wrist elbow",
        cue="Elbows pinned to your sides, curl without swinging, lower slowly."),
    "Dumbbell curl": _e("dumbbell", "biceps", "forearms",
        cue="Palms up, elbows still, squeeze at the top, 2-3 seconds down."),
    "Hammer curl": _e("dumbbell", "biceps", "forearms",
        cue="Palms facing each other, elbows at your sides, no swinging."),
    "Cable curl": _e("cable", "biceps",
        cue="Stand close to the low pulley, elbows fixed, curl and control."),
    "Triceps pushdown": _e("cable", "triceps", "", "elbow",
        cue="Elbows tucked at your sides, push down until your arms are straight; only forearms move."),
    "Overhead dumbbell triceps extension": _e("dumbbell", "triceps", "", "elbow shoulder",
        cue="One dumbbell overhead in both hands, lower behind your head, elbows pointing up."),
    "Skull crusher": _e("barbell bench", "triceps", "", "elbow",
        cue="Lying on the bench, lower the bar toward your forehead by bending only the elbows."),
    "Close-grip push-up": _e("bodyweight", "triceps", "chest", "wrist elbow", True,
        cue="Hands under your shoulders, elbows brushing your ribs as you lower."),
    "Bench dip": _e("bodyweight", "triceps", "shoulders", "shoulder",
        cue="Hands on a sturdy chair or bench behind you, lower until elbows reach 90 degrees."),
    # Legs
    "Barbell back squat": _e("barbell", "quads", "glutes core", "knee lower_back", True,
        cue="Brace, sit down between your heels to at least parallel, knees track over toes."),
    "Goblet squat": _e("dumbbell", "quads", "glutes core", "knee", True,
        cue="Hold a dumbbell at your chest, sit down between your heels, chest up."),
    "Leg press": _e("machine", "quads", "glutes", "knee", True,
        cue="Feet shoulder-width, lower until knees are at about 90 degrees, don't lock out hard."),
    "Leg extension": _e("machine", "quads", "", "knee",
        cue="Pad just above the ankles, straighten your legs and pause, lower slowly."),
    "Bulgarian split squat": _e("dumbbell bench", "quads", "glutes", "knee", True,
        cue="Rear foot on the bench, drop the back knee straight down, front heel stays planted."),
    "Bodyweight split squat (rear foot on a chair)": _e("bodyweight", "quads", "glutes", "knee", True,
        cue="Rear foot on a chair, lower your back knee toward the floor, stay upright."),
    "Dumbbell walking lunge": _e("dumbbell", "quads", "glutes", "knee", True,
        cue="Long steps, back knee just above the floor, push through the front heel."),
    "Reverse lunge": _e("bodyweight", "quads", "glutes", "knee", True,
        cue="Step back, lower the back knee toward the floor, drive through the front heel to return."),
    "Dumbbell step-up": _e("dumbbell bench", "quads", "glutes", "knee", True,
        cue="Whole foot on the bench, stand up through that leg without pushing off the back foot."),
    "Bodyweight squat": _e("bodyweight", "quads", "glutes", "knee", True,
        cue="Feet shoulder-width, sit back and down, heels flat, chest up."),
    "Wall sit": _e("bodyweight", "quads", "glutes", "knee", unit="seconds",
        cue="Back flat on the wall, thighs parallel to the floor, hold."),
    "Standing calf raise": _e("bodyweight", "calves",
        cue="On a step edge, rise onto your toes, pause, lower until your heels drop below the step."),
    "Seated calf raise": _e("machine", "calves",
        cue="Pad on your thighs, full stretch at the bottom, pause at the top."),
    # Core
    "Plank": _e("bodyweight", "core", unit="seconds",
        cue="Elbows under shoulders, body straight, squeeze glutes, don't let your hips sag."),
    "Side plank": _e("bodyweight", "core", unit="seconds",
        cue="Elbow under shoulder, hips lifted in a straight line; do both sides."),
    "Dead bug": _e("bodyweight", "core",
        cue="Lower back pressed to the floor, slowly extend opposite arm and leg."),
    "Bird dog": _e("bodyweight", "core", "glutes",
        cue="On hands and knees, extend opposite arm and leg, keep your hips level."),
    "Hanging knee raise": _e("pullup_bar", "core", "forearms", "shoulder",
        cue="Hang still, bring your knees toward your chest without swinging."),
    "Cable crunch": _e("cable", "core",
        cue="Kneel facing the cable, curl your ribs toward your hips, hips stay still."),
    "Pallof press": _e("cable", "core",
        cue="Stand side-on to the cable, press the handle straight out and resist the twist."),
    # Cardio and conditioning
    "Brisk walk or incline walk": _e("bodyweight", "cardio", unit="minutes",
        cue="A pace where you can talk but not sing."),
    "Incline treadmill walk": _e("machine", "cardio", unit="minutes",
        cue="Incline 6-12%, don't hold the handrails."),
    "Stationary bike": _e("machine", "cardio", unit="minutes",
        cue="Moderate pace; for intervals alternate 30 s hard and 90 s easy."),
    "Rowing machine": _e("machine", "cardio", "back", "lower_back", unit="minutes",
        cue="Legs, then body, then arms on the drive; reverse on the way back."),
    "Mountain climbers": _e("bodyweight", "cardio", "core", "wrist", unit="seconds",
        cue="Plank position, drive your knees toward your chest quickly, hips low."),
    "Jumping jacks": _e("bodyweight", "cardio", "", "knee ankle", unit="seconds",
        cue="Land softly on the balls of your feet."),
    "Burpees": _e("bodyweight", "cardio", "chest quads", "knee wrist lower_back",
        cue="Squat, kick back to a plank, jump your feet in, stand or jump up."),
}


def injury_tags(injuries: str) -> set[str]:
    text = injuries.lower()
    return {tag for pattern, tag in INJURY_PATTERNS if re.search(pattern, text)}


def allowed_exercises(equipment: Equipment, injuries: str) -> list[str]:
    available = EQUIPMENT_AVAILABLE[equipment]
    avoid = injury_tags(injuries)
    return [
        name for name, ex in EXERCISES.items()
        if ex.equipment <= available and not (ex.stresses & avoid)
    ]


def format_amount(name: str, low: int, high: int) -> str:
    unit = EXERCISES[name].unit
    span = f"{low}" if low == high else f"{low}-{high}"
    return {"reps": span, "seconds": f"{span} s", "minutes": f"{span} min"}[unit]

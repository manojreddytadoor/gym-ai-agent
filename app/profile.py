"""The user information both agents build their plans from."""

from typing import Literal

from pydantic import BaseModel, Field

Sex = Literal["male", "female", "other"]
Goal = Literal["fat_loss", "muscle_gain", "recomposition", "strength", "general_fitness", "endurance"]
DailyActivity = Literal["sedentary", "lightly_active", "active"]
Experience = Literal["beginner", "intermediate", "advanced"]
Equipment = Literal["full_gym", "home_gym", "dumbbells_only", "bodyweight_only"]
DietType = Literal["non_vegetarian", "eggetarian", "vegetarian", "vegan", "pescatarian"]
WorkoutTime = Literal["early_morning", "morning", "afternoon", "evening", "night"]
Budget = Literal["low", "moderate", "flexible"]

LABELS = {
    "fat_loss": "Fat loss",
    "muscle_gain": "Muscle gain",
    "recomposition": "Body recomposition (lose fat, gain muscle)",
    "strength": "Strength",
    "general_fitness": "General fitness",
    "endurance": "Endurance",
    "sedentary": "Sedentary (desk job, under ~5k steps/day)",
    "lightly_active": "Lightly active (on feet part of the day, ~5-10k steps/day)",
    "active": "Active (physical job or 10k+ steps/day)",
    "full_gym": "Full commercial gym",
    "home_gym": "Home gym (barbell, rack, bench, dumbbells)",
    "dumbbells_only": "Dumbbells and a bench only",
    "bodyweight_only": "Bodyweight only (no equipment)",
    "non_vegetarian": "Non-vegetarian",
    "eggetarian": "Eggetarian (vegetarian + eggs)",
    "vegetarian": "Vegetarian (dairy, no eggs)",
    "vegan": "Vegan",
    "pescatarian": "Pescatarian",
}


class UserProfile(BaseModel):
    name: str = Field(default="", max_length=60)
    age: int = Field(ge=16, le=90)
    sex: Sex
    height_cm: float = Field(ge=120, le=230)
    weight_kg: float = Field(ge=35, le=250)
    target_weight_kg: float | None = Field(default=None, ge=35, le=250)
    goal: Goal
    daily_activity: DailyActivity
    experience: Experience
    training_days_per_week: int = Field(ge=2, le=7)
    session_minutes: int = Field(ge=20, le=150)
    workout_time: WorkoutTime
    equipment: Equipment
    injuries: str = Field(default="", max_length=500)
    medical_conditions: str = Field(default="", max_length=500)
    diet_type: DietType
    allergies: str = Field(default="", max_length=300)
    cuisine: str = Field(default="", max_length=200)
    meals_per_day: int = Field(ge=2, le=6)
    budget: Budget = "moderate"
    notes: str = Field(default="", max_length=1000)

    def to_prompt(self) -> str:
        """Render the profile as a tagged block for the agents' first message."""

        def label(value: str) -> str:
            return LABELS.get(value, value.replace("_", " "))

        rows = [
            ("Name", self.name or "not given"),
            ("Age", self.age),
            ("Sex", self.sex),
            ("Height", f"{self.height_cm:g} cm"),
            ("Current weight", f"{self.weight_kg:g} kg"),
            ("Target weight", f"{self.target_weight_kg:g} kg" if self.target_weight_kg else "not given"),
            ("Primary goal", label(self.goal)),
            ("Daily activity outside workouts", label(self.daily_activity)),
            ("Training experience", self.experience),
            ("Training days per week", self.training_days_per_week),
            ("Time per session", f"{self.session_minutes} minutes"),
            ("Usual workout time", label(self.workout_time)),
            ("Equipment", label(self.equipment)),
            ("Injuries / physical limitations", self.injuries or "none reported"),
            ("Medical conditions", self.medical_conditions or "none reported"),
            ("Diet type", label(self.diet_type)),
            ("Allergies / foods to avoid", self.allergies or "none reported"),
            ("Cuisine preference", self.cuisine or "no preference"),
            ("Meals per day", self.meals_per_day),
            ("Food budget", self.budget),
            ("Other notes", self.notes or "none"),
        ]
        body = "\n".join(f"- {key}: {value}" for key, value in rows)
        return f"<user_profile>\n{body}\n</user_profile>"

# Gym AI Coaches

Two AI agents that create personalized plans from a user's profile, running entirely on your own computer with [Ollama](https://ollama.com). No API key and no cost:

- **Diet Coach**: calorie and macro targets, a full-day meal plan with exact portions, swaps and tips.
- **Workout Coach**: a weekly program with sets, reps, rest and effort for each exercise, a weekly volume check, progression and recovery guidance.

Both coaches work at the same time, and you can ask follow-ups like "swap paneer for tofu" or "I can't do squats, my knee hurts".

## How the results are kept trustworthy

Small local models are good at choosing sensible foods and exercises, but unreliable with numbers and rules. For example, `llama3.2` once suggested 250 g of dry oats for breakfast. So the work is split:

| The AI model decides | The app guarantees |
|---|---|
| Which foods go in each meal, and dish names | Foods come only from a database of 82 foods with USDA / IFCT values, filtered for the user's diet type and allergies before the model sees them |
| Rough portions | Portions are capped at sensible amounts and fitted automatically so calories land within 5% of target and protein is met |
| Which exercises each day, and how many sets | Exercises come only from a list of 78, filtered for the user's equipment and injuries |
| The split and each day's focus | Reps, rest, effort (RIR), warm-ups, training days, progression and recovery advice are set from the goal and exercise type; session length and weekly sets per muscle are calculated, and over-long sessions are trimmed |

The model's output is constrained to a JSON schema with Ollama's structured outputs, so it can't pick anything outside the filtered lists. After each draft, the app checks the plan. Problems it can't fix itself, such as missing protein foods or an untrained muscle group, go back to the model for revision, up to two rounds.

Because a small model on a CPU writes only about 5 tokens per second, the workout coach builds the week **one day at a time** rather than in a single long answer. Each call is short, progress is visible ("Building day 3 of 5"), and the model only has to name exercises and sets.

The page header shows the exact model and its ID, with a link to compare it against the official listing on ollama.com.

```
app/
  profile.py     user profile model (the form fields)
  nutrition.py   calorie/macro targets, food database, diet/allergy filtering, portion fitting
  exercises.py   exercise database, equipment/injury filtering
  training.py    session length and weekly-volume analysis
  llm.py         Ollama client with JSON-schema output
  coaches.py     the Diet and Workout coaches: prompts, checks, revisions, rendering
  server.py      FastAPI app: creates sessions and streams progress
  static/        the web page
tests/           unit tests and coach-flow tests with a scripted model
```

## Setup

1. Install [Ollama](https://ollama.com/download) and download a model:

   ```bash
   ollama pull llama3.2
   ```

2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the server from this folder, then open http://127.0.0.1:8000:

   ```bash
   python -m app.server
   ```

On a laptop without a dedicated graphics card, a plan takes a few minutes.

## Choosing a model

Set `OLLAMA_MODEL` before starting the server to use another model:

| Model | Download | Notes |
|---|---|---|
| `llama3.2` (default) | 2.0 GB | Fastest. Plans may need a revision round. |
| `granite4.1:8b` | 5.3 GB | IBM model built for structured output. Better choices, about 2-3x slower on a CPU. |

```bash
ollama pull granite4.1:8b
```

Command Prompt: `set OLLAMA_MODEL=granite4.1:8b`. PowerShell: `$env:OLLAMA_MODEL = "granite4.1:8b"`. `OLLAMA_HOST` points to an Ollama server on another machine.

## Tests

```bash
python -m pytest
```

The tests use a scripted stand-in for the model, so they run in seconds without Ollama.

## Limitations

- Sessions are kept in memory. Restarting the server clears them.
- Food values are approximate, and packaged foods and recipes vary.
- Injury filtering avoids exercises that load the named joint. It is not a substitute for a physiotherapist.
- Plans are general guidance, not medical advice.

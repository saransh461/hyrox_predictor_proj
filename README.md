# Hyrox Performance Analysis & Finish Time Predictor

> Analyzing 90,000+ Hyrox race results to understand what actually drives performance, and building models and an interactive app to predict finish times and athlete archetypes.

**[Try the live app →](#)** *(add link once deployed)*

## Why this project

I've been training in the gym for a few years and have recently gotten into Hyrox and endurance racing content. I wanted to understand, with real data, what actually separates fast finishers from the rest — is it running speed, station strength, or something less obvious like transition efficiency?

## Dataset

- Source: [Kaggle — Hyrox Results Dataset](https://www.kaggle.com/datasets/jgug05/hyrox-results)
- ~90,000 race entries, filtered to solo (Open/Pro) divisions for individual performance analysis
- Each entry includes 8 station splits (work), 8 running segments (run), and 8 transition times (roxzone)
- Cleaning: removed invalid zero-second finish times, team/relay divisions, and broken station splits (>900s on a single station — likely data errors or race incidents)

## Key findings

- **Roxzone (transition) time is highly consistent across athletes** and shows the least variability of any race component — it doesn't meaningfully separate fast and slow finishers.
- **Running time shows the most variability between athletes** and has the strongest individual correlation with total finish time of the three race components.
- **Gender shows minimal difference** in average running performance in this dataset.
- **Running performance declines gradually with age** — from ~2674s (30-34 group) to ~2904s (55-59 group). The trend becomes statistically unreliable beyond age 60-64 due to small sample sizes (verified via standard error, not just raw counts).
- **Station (work) performance degrades roughly twice as fast as running with age** — work time increased ~17.8% from the 30-34 to 55-59 groups, vs an 8.6% increase in running time. Suggests strength/power-based movements decline faster with age than aerobic endurance.
- **Station consistency is the strongest performance predictor found** — athletes with more even performance across all 8 stations finish faster overall (r ≈ 0.71), more strongly than age, gender, or run/work balance individually.

## Features engineered

- `run_to_work_ratio` — relative balance between running speed and station performance
- `pacing_slope` — linear trend across the 8 running splits; measures whether an athlete slows down over the race (fatigue)
- `station_consistency` — standard deviation across the 8 station times; measures how evenly an athlete performs vs having major weak points. Strongest engineered feature (r ≈ 0.71 with finish time)
- `proxy_consistency` — simplified version of `station_consistency` using only fastest/slowest station time (2 inputs instead of 8); validated at r = 0.96 against the full measure, used in the app for a lighter input form

## Modelling

Two models were built deliberately, answering two different questions:

| | Model A (explanatory) | Model B (pre-race) |
|---|---|---|
| Question | What explains a result, given race data? | What's the most honest prediction possible *before* racing? |
| Inputs | Age, gender, station consistency, pacing slope, run/work ratio, roxzone time | Age, gender only |
| MAE | 300s (XGBoost) / 329s (Linear Regression) | 731s |

**Why two models:** Model A uses information only available *after* a race (e.g. pacing patterns, station consistency). Using it to claim "predict your time before racing" would be misleading, since nobody knows their pacing_slope before they've raced. Model B uses only what's realistically known in advance — its weaker accuracy (731s vs 300s) is itself a finding: it quantifies how much of Hyrox performance isn't explained by basic demographics alone.

**Linear Regression → XGBoost:** switching models reduced MAE by ~9% (329s → 300s), but the accuracy gap between faster and slower athletes stayed nearly unchanged (~90s in both). This suggests the limitation isn't model complexity — it's missing information. Available features explain top-finisher performance well but don't fully capture what drives variability for recreational-pace athletes; closing this gap would likely need additional data (training history, strength benchmarks).

## Athlete archetypes (K-means clustering)

Clustered athletes on `run_to_work_ratio` and `station_consistency` (K=3, chosen after comparing against K=4, which only split existing groups into finer gradations rather than revealing new patterns):

- **Balanced & Consistent** — even performance across stations, fastest average finish time (~84 min)
- **Runner-Leaning** — relatively stronger running vs station performance, still highly consistent, second-fastest (~89 min)
- **Inconsistent** — uneven station performance regardless of run/work balance, clearly the slowest group (~106 min)

Station consistency matters more for overall performance than run/work balance itself — both fast-finishing groups share low `station_consistency`, while the slow group is defined primarily by unevenness, not a specific run/station weakness.

## The app

An interactive Streamlit app lets users enter a small set of inputs (age, gender, total run/work time, fastest/slowest station time) and get:
- A predicted finish time
- Their percentile vs. the real dataset
- Their athlete archetype, with a short explanation

**Model A-lite:** the app uses a simplified model trained on only the inputs it can realistically collect (`age_group`, `gender`, `total_run_secs`, `run_to_work_ratio`, `proxy_consistency`) — MAE 87s. This is more accurate than the original Model A despite fewer features, largely because `total_run_secs` is a direct, large component of total finish time, giving the model a strong anchor.

**A bug worth noting:** an earlier version of Model A-lite (without `total_run_secs`) produced physically impossible predictions — e.g., a predicted total time less than the user's own input run+work time. Root cause: the model had no feature capturing absolute pace, only relative ratios, so two athletes with very different speeds but the same ratio were treated identically. Fixed by adding `total_run_secs` as a direct input (MAE improved 462s → 87s).

**A second limitation found during testing:** at the extremes of the input range (e.g. very fast running times), predictions became unreliable again. Investigation showed only ~0.35% of athletes in the training data fall in that range — too sparse for the model (XGBoost's tree-based splits don't interpolate smoothly between sparse points) to learn a reliable pattern. Fixed by restricting the app's input range to the well-supported 5th–95th percentile of the real data.

## Repo structure

```
notebooks/
  01_eda.ipynb          # cleaning, EDA, feature engineering
  02_modelling.ipynb    # Model A, Model B, Model A-lite, clustering
app/
  streamlit_app.py      # interactive prediction + archetype app
  *.pkl                 # saved trained models (model, kmeans, scaler)
  all_finish_times.csv  # lookup table for percentile calculation
data/
  (not included — see below)
```

## How to run this

```bash
git clone https://github.com/saransh461/hyrox_predictor_proj.git
cd hyrox_predictor_proj
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jgug05/hyrox-results) and place it in a `data/` folder before running the notebooks.

To run the app:
```bash
cd app
streamlit run streamlit_app.py
```

## What's next

- Add a pre-race tab to the app using Model B, clearly separated from the post-race analysis
- Explore additional data sources (training history, strength benchmarks) to close the fast/slow accuracy gap in Model A
- Extend the analysis to Ironman results for a cross-discipline pacing comparison
- Deploy the app on Streamlit Community Cloud for public access
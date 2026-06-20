# Hyrox Performance Analysis & Finish Time Predictor

> Analyzing 90,000+ Hyrox race results to understand what actually drives performance — and building a model to predict finish times.

## Why this project

I've been training in the gym for a few years and have recently gotten into Hyrox and endurance racing content. I wanted to understand, with real data, what actually separates fast finishers from the rest — is it running speed, station strength, or something less obvious like transition efficiency?

## Dataset

- Source: [Kaggle — Hyrox Results Dataset](https://www.kaggle.com/datasets/jgug05/hyrox-results)
- ~90,000 race entries, filtered down to solo (Open/Pro) divisions for individual performance analysis
- Each entry includes 8 station splits (work), 8 running segments (run), and 8 transition times (roxzone)

## Key findings so far

- **Transition (roxzone) time is consistent across athletes** and shows the least variability of any race component — it's not a meaningful differentiator between fast and slow finishers.
- **Running time shows the most variability** between athletes and has the strongest correlation with total finish time.
- **Gender shows minimal difference** in average running performance in this dataset.
- **Running performance declines gradually with age**, from ~2674s (30-34 group) to ~2904s (55-59 group) — though the trend becomes statistically unreliable beyond age 60-64 due to small sample sizes.
- **Work Times degrade twice as fast compared to running**,  work time increased ~17.8% from the 30-34 to 55-59 age groups, compared to an 8.6% increase in running time. This suggests strength/power-based movements degrade faster with age than aerobic endurance
- **Station consistency is a strong predictor of overall performance** — 
  athletes with more even performance across all 8 stations tend to finish 
  faster overall (r ≈ 0.71), more strongly than most other individual factors examined.

## Features engineered

- `run_to_work_ratio` — relative balance between running speed and station performance
- `pacing_slope` — measures whether an athlete slows down over the course of the race (fatigue)
- `station_consistency` — standard deviation across the 8 station times for each athlete; measures how evenly they perform across all stations vs having major weak points. Strongest engineered feature so far — correlates ~0.71 with total finish time.

## Model A — Explanatory model

Predicts finish time using age, gender, and engineered features (station 
consistency, pacing slope, run-to-work ratio, transition time). This model 
uses information only available *after* a race, so it answers "what factors 
explain performance" rather than predicting a time before racing.

**Baseline: Linear Regression**
- MAE: 329 seconds (~5.5 minutes), about 6% of average finish time
- Accuracy is uneven across the field: MAE for the faster half of athletes 
  is ~90 seconds lower than for the slower half — the model explains 
  top-finisher performance more reliably than recreational-pace performance, 
  likely because fast finishers' results are more directly driven by 
  consistency and pacing, while slower finishers show more variability the 
  current features don't capture.

Next: comparing against XGBoost to see if a non-linear model closes this gap.

**XGBoost**
Switching from Linear Regression to XGBoost reduced overall MAE by ~9% (329s → 300s), but the gap in accuracy between faster and slower athletes remained nearly unchanged (~90s in both models). This suggests the limitation isn't model complexity, but missing information — the available features (age, gender, pacing/consistency ratios) explain top-finisher performance well, but don't fully capture what drives variability for recreational-pace athletes. Additional data (e.g., training history, strength benchmarks) would likely be needed to close this gap

**Model B: Pre-race predictor (honest, limited inputs)**
- Inputs: only age group and gender — the only information realistically 
  knowable before an athlete has raced
- MAE: 731 seconds (~12.2 minutes), notably worse than Model A
- This is an expected and informative result: it quantifies just how much 
  of Hyrox performance is *not* explained by basic demographics alone, 
  reinforcing the EDA finding that age and gender have only modest individual 
  effects. A genuinely useful pre-race predictor would require additional 
  inputs not present in this dataset — e.g., training history, prior race 
  times, or strength benchmarks.

  ## Athlete archetypes (K-means clustering)

Clustered athletes on `run_to_work_ratio` and `station_consistency` (K=3, 
chosen after comparing against K=4, which only split existing groups into 
finer gradations rather than revealing new patterns):

- **Balanced & Consistent** — even performance across stations, fastest 
  average finish time (~84min)
- **Runner-Leaning** — relatively stronger running vs station performance, 
  still highly consistent, second-fastest group (~89min)
- **Inconsistent** — uneven station performance regardless of run/work 
  balance, clearly the slowest group (~106min)

Station consistency appears to matter more for overall performance than 
the run/work balance itself — both fast-finishing groups (Balanced and 
Runner-Leaning) share low station_consistency, while the slow group is 
defined primarily by unevenness, not a particular run/station weakness.

## App design note: simplified consistency proxy

The full `station_consistency` feature (std across 8 stations) requires 8 
individual inputs, which adds friction for app users. Tested a simplified 
proxy — `slowest_station - fastest_station` (2 inputs) — and found it 
correlates 0.96 with the full measure, making it a safe simplification 
for the app's lightweight input design.

## What's in this repo

```
notebooks/
  01_eda.ipynb              # exploratory analysis
  02_feature_engineering.ipynb   (coming soon)
  03_modelling.ipynb             (coming soon)
app/
  streamlit_app.py          (coming soon)
```

## How to run this

```bash
git clone https://github.com/saransh461/hyrox_predictor_proj.git
cd hyrox_predictor_proj
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the dataset from [Kaggle link] and place it in a `data/` folder before running the notebooks.

## What's next

- Feature engineering: station consistency scores, run-to-work ratios
- Build a finish time predictor (XGBoost) based on pre-race inputs
- Cluster athletes into performance archetypes (K-means)
- Deploy as an interactive Streamlit app

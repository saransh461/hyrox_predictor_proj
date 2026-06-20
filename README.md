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

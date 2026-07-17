# Cookie Cats — Gate Position A/B Test Analysis

An end-to-end data science project analysing a mobile game A/B test, from exploratory data analysis through A/B testing and business impact simulation.

[![Python](https://img.shields.io/badge/Python-3.13-7C3AED?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Dash-F59E0B?style=flat-square&logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![uv](https://img.shields.io/badge/uv-package_manager-10B981?style=flat-square)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

---

## The Question

> *Should the first progression gate in Cookie Cats be placed at level 30 or level 40 to maximise 7-day player retention?*

In casual mobile games, **progression gates** are design elements that introduce friction: players must wait or pay to continue. Placed too early, they cause churn. Placed too late, they lose their engagement-refreshing effect. This project evaluates which position works better using rigorous statistical methods.

---

## Key Results


| Metric | Gate 30 | Gate 40 | Bootstrap P(30>40) | Z-statistic | p-value | Chi-Square | p-value |  Cohen's h | Effect Size |
|---|---|---|---|---|---|---|---|---|---|
| Day-1 Retention | ~44.8% | ~44.23% | ~96.8% | 1.787 | ~0.03 | 3.1698 | ~0.08 | ~0.012 | Small |
| Day-7 Retention | ~19.0%| ~18.20% | ~99.9% | 3.157 | ~0.0007 | 9.9153 | ~0.002 | ~0.021 | Small |


**Recommendation: keep the gate at level 30.**  
Four independent methods (bootstrap, z-test, chi-square, Cohen's h) all converge on the same conclusion.

---

## Project Structure

```
cookie-cats-ab-analysis/
│
├── data/
│   ├── raw/                        # Original dataset (unmodified)
│   └── processed/                  # Cleaned dataset with segment column
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis & data cleaning
│   └── 02_ab_testing.ipynb         # Frequentist hypothesis testing
│
├── scripts/
│   └── 01_eda.py                   # Script version of the EDA notebook (Auto-generated from 01_eda.ipynb)
│
├── dashboard/
│   ├── app.py                      # Plotly Dash interactive dashboard
│   └── requirements.txt
│
├── reports/
│   └── executive_summary.md        # Non-technical summary & recommendation
│
├── pyproject.toml                  # Dependencies managed with uv
├── uv.lock
└── README.md
```

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Data quality validation, engagement distribution, player segmentation, retention baseline |
| 02 | [`02_ab_testing.ipynb`](notebooks/02_ab_testing.ipynb) | Bootstrap simulation, z-test, chi-square, Cohen's h effect size |


---

## Dashboards

| Tool | Link |
|---|---|
| 📊 Plotly Dash | *Run Locally* |


---

## Methodology

### Frequentist (Notebook 02)
- **Bootstrap simulation** — 10,000 iterations to estimate the sampling distribution of retention lift without parametric assumptions
- **Z-test for proportions** — one-sided test with 95% Wilson confidence intervals
- **Chi-square test** — independence test to validate z-test findings via an alternative framework
- **Cohen's h** — effect size measure to distinguish statistical from practical significance

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/NataliaBackhaus/game-retention-cookiecats.git
cd game-retention-cookiecats

# Download dataset
curl -L -o mobile-games-ab-testing-cookie-cats.zip https://www.kaggle.com/api/v1/datasets/download/mursideyarkin/mobile-games-ab-testing-cookie-cats
mkdir -p data/raw
unzip -j mobile-games-ab-testing-cookie-cats.zip -d data/raw
rm mobile-games-ab-testing-cookie-cats.zip

# Install dependencies
uv sync

# Run the dashboard (to run the dashbord, you need the data cleaned. So, before running this, run the notebook/01_eda.ipynb)
# Alternatively, you can just run 'cd scripts/ && uv run python 01_eda.py' (automatically generatad from notebooks/01_eda.ipynb) to generate the clean data.
uv run python dashboard/app.py
```

**Main dependencies:** `pandas`, `numpy`, `scipy`, `statsmodels`, `pymc`, `arviz`, `plotly`, `dash`

---

## Dataset

**Source:** [Mobile Games A/B Testing — Cookie Cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats) (Kaggle)  
**Players:** 90,189  
**Columns:** `userid`, `version`, `sum_gamerounds`, `retention_1`, `retention_7`

The dataset captures player behaviour for 14 days post-install. One extreme outlier (49,854 rounds) was removed prior to analysis (likely a bot or automated testing account).

---

## What I Would Do With More Data

- **Link IAP events to player IDs** to replace proxy revenue estimates with actual ARPU metrics
- **Extend the observation window** to Day-14 and Day-30 to assess whether the retention advantage persists over the full player lifecycle
- **Build a hierarchical Bayesian model** to estimate gate effects per segment simultaneously, enabling personalised gate positioning
- **Test subsequent gate positions**: this analysis covers only the first gate; optimal positions for later gates remain an open question

---

## About

This project was built as part of a data science portfolio focused on the games industry.  
For the non-technical summary, see [`reports/executive_summary.md`](reports/executive_summary.md).


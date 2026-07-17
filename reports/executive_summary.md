# Executive Summary: Cookie Cats Gate Position A/B Test

## 1. Project Overview

Cookie Cats is a mobile puzzle game in the "connect-three" genre. To progress through the game, players periodically encounter **gates** mandatory pauses that require them to wait or make an in-app purchase to continue.

This project analyses an A/B test in which the position of the first gate was moved from **level 30** (control group) to **level 40** (treatment group). The experiment covered **90,189 players** split approximately 50/50 between the two variants.

The central business question: *does moving the gate later in the game improve or harm player retention?*

---

## 2. Key Results

| Metric | Gate 30 | Gate 40 | Bootstrap P(30>40) | Z-statistic | p-value | Chi-Square | p-value |  Cohen's h | Effect Size |
|---|---|---|---|---|---|---|---|---|---|
| Day-1 Retention | ~44.8% | ~44.23% | ~96.8% | 1.787 | ~0.03 | 3.1698 | ~0.08 | ~0.012 | Small |
| Day-7 Retention | ~19.0%| ~18.20% | ~99.9% | 3.157 | ~0.0007 | 9.9153 | ~0.002 | ~0.021 | Small |

Day-1 retention shows a statistically detectable but practically negligible difference between groups. Day-7 retention, the metric most directly influenced by the gate, as it captures players who have progressed far enough to encounter it, shows a consistent and robust advantage for gate_30 across all methods applied.

---

## 3. Recommendation

> **Keep the gate at level 30.**

The evidence is consistent across five independent statistical methods. Gate_30 produces higher 7-day retention with a posterior probability exceeding 99%, meaning the observed advantage is almost certainly not due to chance.

The mechanism is consistent with the **hedonic adaptation** hypothesis: an earlier forced pause refreshes player engagement, increasing the likelihood of a return visit over the following week. Players who are made to wait at level 30 return more often than those who encounter the gate later, or not at all within their first week.

---

## 5. Methodology

The analysis was conducted across four notebooks:

| Notebook | Contents |
|---|---|
| [`01_eda.ipynb`](../notebooks/01_eda.ipynb) | Data quality validation, engagement distribution, retention baseline, player segmentation |
| [`02_ab_testing.ipynb`](../notebooks/02_ab_testing.ipynb) | Bootstrap simulation, z-test for proportions, chi-square test, Cohen's h effect size |


All five statistical methods applied (bootstrap, z-test, chi-square) converge on the same conclusion, substantially reducing the risk of a false positive driven by any single method's assumptions.

---

## 6. Limitations


- **Short observation window.** The experiment captures Day-1 and Day-7 retention only. Day-14 and Day-30 effects are unknown and could differ.
- **Uniform treatment.** The analysis treats all players as a single population. The gate effect may differ across engagement segments; a personalised gate position could yield stronger outcomes.

---

## 7. Next Steps

- **Immediate:** Deploy gate_30 to 100% of new players via feature flag.
- **Short-term:** Instrument in-app purchase events per player to enable revenue-linked analysis in future experiments.
- **Short-term:** Run a segment-level follow-up experiment to assess whether Hardcore players respond differently to gate positions between levels 30 and 40.
- **Long-term:** Build a hierarchical Bayesian model to estimate segment-level gate effects simultaneously and explore personalised gate positioning.

---

*For full methodology, code, and statistical outputs, refer to the notebook series linked in Section 5.*

#!/usr/bin/env python
# coding: utf-8

# # Cookie Cats: Exploratory Data Analysis (EDA) & Data Cleaning

# ## 1. Introduction
# 
# This project analyzes an A/B test conducted in **Cookie Cats**, a popular mobile puzzle game in the "connect-three" genre.
# 
# The objective of the experiment is to evaluate the impact of moving the first progression gate from Level 30 to Level 40, and its effect on player retention and engagement.
# 
# In casual mobile games, **progression gates** are critical design elements that introduce friction by requiring players to wait or make in-app purchases to continue. 
# 
# While this friction can support monetization and long-term engagement, poorly positioned gates may negatively impact the player experience, leading to increased churn.
# 
# This analysis aims to determine whether shifting the gate improves or harms key performance indicators, ultimately supporting a data-driven product decision.
# 
# 
# 
# ## 2. The Dataset
# 
# The dataset contains behavioral data from 90,189 players who installed the game during the experiment period.
# 
# Each player was randomly assigned to one of two variants:
# - **gate_30 (control group):** gate placed at level 30  
# - **gate_40 (treatment group):** gate placed at level 40  
# 
# ### Data Dictionary:
# 
# - **userid**: Unique identifier for each player  
# - **version**: Experiment group assignment (gate_30 or gate_40)  
# - **sum_gamerounds**: Total number of game rounds played within the first 14 days  
# - **retention_1**: Whether the player returned 1 day after installation  
# - **retention_7**: Whether the player returned 7 days after installation  
# 
# 
# 
# ## 3. EDA Objectives
# 
# Before conducting statistical hypothesis testing, we perform an exploratory data analysis to validate data quality and establish a reliable foundation for the experiment evaluation.
# 
# The main objectives of this stage are:
# 
# 1. **Data Quality Validation**  
#    Ensure the dataset is complete, consistent, and free of anomalies that could bias the experiment results.
# 
# 2. **Player Behavior Analysis**  
#    Understand how players interact with the game, particularly in terms of engagement distribution and activity patterns.
# 
# 3. **Retention Baseline Definition**  
#    Establish baseline retention metrics (Day 1 and Day 7) to contextualize the impact of the experiment.
# 
# By completing these steps, we ensure that the A/B test results can be interpreted with confidence and translated into actionable product insights.
# 
# The final goal is to provide a clear recommendation on whether the new gate position should be adopted in production.
# 
# ---

# ### **Setup**

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set default Plotly theme
pio.templates.default = 'plotly_white'

# Define a consistent color palette for A/B groups across all visualizations
COLOR_MAP = {'gate_30': '#7C3AED', 'gate_40': '#F59E0B'}

# Read dataset
df = pd.read_csv('../data/raw/cookie_cats.csv')


# ---

# ### **Data Quality Validation**

# Before proceeding with the experiment analysis, we validate data quality and integrity to ensure that any observed differences between groups can be reliably attributed to the experimental treatment, rather than artifacts introduced by data collection or processing. We begin by verifying the structural integrity of the dataset, including column consistency, and data types

# In[2]:


print("--- Dataset Info ---")
print(df.info())
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Dataset Head ---")
df.head()


# The absence of missing values suggests a reliable telemetry pipeline, reducing the risk of biased retention estimates due to incomplete user tracking. We must also check for duplicates and for the uniqueness of user identifiers

# In[3]:


# Check for duplicate rows
n_duplicate_rows = df.duplicated().sum()
print(f'Duplicate rows: {n_duplicate_rows}')

# Confirm that each userid is unique: one entry per player
n_unique_users = df['userid'].nunique()
print(f'Unique user IDs: {n_unique_users} out of {len(df)} rows')


# No data quality issues were identified, indicating that the dataset is complete and reliable.
# 
# This suggests that:
# - Player activity has been consistently tracked
# - The telemetry pipeline is stable
# - Retention metrics can be interpreted with a lower risk of measurement bias
# 
# With data quality validated, we now examine the distribution of total game rounds played (`sum_gamerounds`) to identify extreme player behavior that could influence aggregate metrics.

# We also verify that the A/B groups are balanced in size, as a significant imbalance could indicate a randomization failure and undermine the validity of downstream statistical tests

# In[4]:


# Confirm group sizes and balance
group_counts = df['version'].value_counts().reset_index()
group_counts.columns = ['version', 'count']
group_counts['share (%)'] = (group_counts['count'] / len(df) * 100).round(2)
print(group_counts)

# The two groups are approximately equal in size (~50/50 split), confirming that randomization was executed correctly. This balance is a prerequisite for valid A/B test comparisons

# #### Outlier Detection and Treatment
# 
# We analyze the distribution of total game rounds played (`sum_gamerounds`) to identify extreme player behavior.
# 
# Outliers in this context may represent:
# - Highly engaged users (power players)
# - Automated behavior (bots)
# - Data anomalies
# 
# These extreme values are important because they can disproportionately influence aggregate metrics, such as mean engagement, and potentially distort the interpretation of the experiment results

# In[5]:


# In[6]:


# Identify the most extreme value and calculate the 99th percentile threshold
max_rounds = df['sum_gamerounds'].max()
p99 = df['sum_gamerounds'].quantile(0.99)

print(f'Maximum rounds observed: {max_rounds:,}')
print(f'99th percentile: {p99:.0f} rounds')


# A single player with nearly 50,000 rounds was identified: approximately 17x above the 99th percentile. This is a strong candidate for a bot or an automated testing account rather than an organic user.
# 
# **Treatment strategy:** We remove only this single extreme observation from the working dataset. All other players, including those at the 99th percentile, are retained to preserve statistical power. The original raw data remains unchanged in `data/raw/`

# In[7]:


# Remove the single extreme outlier (the maximum value only)
df = df[df['sum_gamerounds'] < max_rounds].copy()

print(f'Rows removed: 1 (userid with {max_rounds:,} rounds)')
print(f'Remaining rows: {len(df):,}')

# ---

# ### **Player Behavior Analysis**

# With the dataset validated, we examine the distribution of player engagement (measured by total game rounds played) to understand behavioral patterns across the player base

# In[8]:


# Summary statistics for game rounds: overall and per group
print('--- Overall Engagement Metrics ---')
print(df['sum_gamerounds'].describe().round(2))

print('\n--- Engagement Metrics per A/B Group ---')
print(df.groupby('version')['sum_gamerounds'].agg(['count', 'mean', 'median', 'std', 'max']).round(2))


# The median (16 rounds) is substantially lower than the mean (~51 rounds), confirming a right-skewed distribution, a common pattern in mobile games where the majority of players engage minimally before churning. This skew makes the median a more robust central tendency measure than the mean for this dataset

# In[9]:


# Quantify players who never played a single round (zero-round churn)
# These users installed the game but did not engage at all
non_starters = (df['sum_gamerounds'] == 0).sum()
pct_non_starters = non_starters / len(df) * 100

print(f'Non-starters (0 rounds played): {non_starters:,} ({pct_non_starters:.2f}%)')


# Approximately 4.4% of users never played a single round. This level of zero-round churn is within an expected range for mobile games and does not indicate a systemic onboarding issue. These players are retained in the analysis, as their absence from gameplay is itself a behavioral signal that should be captured in retention metrics

# In[10]:


# Histogram of game rounds: capped at 100 to focus on the core player population
# Gate positions are overlaid to contextualize where most players stop relative to the experiment
df_hist = df[df['sum_gamerounds'] <= 100]

# The distributions are visually similar across both groups, reinforcing confidence in the randomization process. Notably, the majority of players do not reach either gate position (levels 30 or 40), meaning the A/B test primarily captures the behavior of the more engaged segment of the player base

# In[12]:


# Quantify the proportion of players who actually reached each gate threshold
total = len(df)
reached_30 = (df['sum_gamerounds'] >= 30).sum()
reached_40 = (df['sum_gamerounds'] >= 40).sum()

print(f'Players who reached Level 30: {reached_30:,} ({reached_30/total*100:.2f}%)')
print(f'Players who reached Level 40: {reached_40:,} ({reached_40/total*100:.2f}%)')


# ### Player Segmentation by Engagement

# To enable more granular analysis, we segment players into three engagement tiers based on total rounds played. 
# These thresholds are informed by the distribution observed earlier and by common industry conventions for casual mobile games
# 
# | Segment | Rounds played | Profile |
# |---|---|---|
# | Casual | 0 – 10 | Installed and briefly tried the game |
# | Regular | 11 – 50 | Moderate engagement; approaching the gate range |
# | Hardcore | 51+ | High engagement; most likely to interact with the gate |

# In[13]:


# Assign engagement segments based on rounds played
# Thresholds are exploratory and should be revisited with domain input
def assign_segment(rounds):
    if rounds <= 10:
        return 'Casual'
    elif rounds <= 50:
        return 'Regular'
    else:
        return 'Hardcore'

df['segment'] = df['sum_gamerounds'].apply(assign_segment)

# Segment distribution
segment_dist = df['segment'].value_counts().reset_index()
segment_dist.columns = ['segment', 'count']
segment_dist['share (%)'] = (segment_dist['count'] / len(df) * 100).round(2)
print(segment_dist)


# In[14]:


# Visualize segment distribution as a pie chart
SEGMENT_COLORS = {'Casual': '#7C3AED', 'Regular': '#F59E0B', 'Hardcore': '#10B981'}
SEGMENT_ORDER = ['Hardcore', 'Regular', 'Casual']  # ascending order for readability

# The majority of players fall into the Casual segment, consistent with the right-skewed distribution observed earlier. 
# Hardcore players represent a small but strategically important cohort, as they are most likely to encounter (and be affected by) the gate

# ### Retention by Segment and A/B Group

# We disaggregate retention rates by both player segment and A/B group to assess whether the gate position has a differential effect across engagement tiers. 
# This analysis can reveal whether the treatment disproportionately impacts casual or hardcore players

# In[15]:


# Retention rates by segment and A/B group
segment_retention = (
    df.groupby(['segment', 'version'])[['retention_1', 'retention_7']]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)
print(segment_retention)


# The retention gap between groups appears more pronounced among Regular and Hardcore players (those who engage beyond the first few rounds).
# This pattern is consistent with the hypothesis that the gate position primarily affects players who progress far enough to encounter it. 
# The statistical significance of these differences will be evaluated in the next notebook

# ### Conditional Retention: Players Who Reached the Gate

# The overall retention metrics include players who never reached either gate, which dilutes the measured effect of the experiment. 
# To isolate the treatment impact more precisely, we recalculate retention rates considering only players who progressed far enough to encounter the gate assigned to their group.
# 
# This conditional analysis reflects a more realistic view of the gate's influence on player behavior

# In[17]:


# Filter each group to include only players who reached their respective gate
reached_gate = df[
    ((df['version'] == 'gate_30') & (df['sum_gamerounds'] >= 30)) |
    ((df['version'] == 'gate_40') & (df['sum_gamerounds'] >= 40))
]

cond_retention = (
    reached_gate.groupby('version')[['retention_1', 'retention_7']]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)

print(f'Players who reached their gate: {len(reached_gate):,} ({len(reached_gate)/len(df)*100:.1f}% of total)')
print()
print('--- Conditional Retention Rates (%) ---')
print(cond_retention)


# In[18]:


# Compare overall vs conditional retention side by side
overall = (
    df.groupby('version')[['retention_1', 'retention_7']]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)
overall['scope'] = 'All players'
cond_retention['scope'] = 'Players who reached gate'

comparison = pd.concat([overall, cond_retention])
comparison_melted = comparison.melt(
    id_vars=['version', 'scope'],
    value_vars=['retention_1', 'retention_7'],
    var_name='metric',
    value_name='retention_rate'
)
comparison_melted['metric'] = comparison_melted['metric'].map({
    'retention_1': 'Day-1', 'retention_7': 'Day-7'
})
comparison_melted['label'] = comparison_melted['version'] + ' | ' + comparison_melted['scope']

# Among players who reached the gate, the retention difference between groups is more pronounced than in the overall population. 
# This suggests that the gate position has a meaningful effect on the players directly exposed to it, and that overall metrics may underestimate the true impact of the experimental treatment

# ---

# ### **Retention Baseline Definition**

# We now establish baseline retention metrics for Day 1 and Day 7, disaggregated by A/B group. These figures will serve as the primary outcome variables in the subsequent hypothesis testing notebook

# In[19]:


# Overall retention rates across the full player base
overall_ret_1 = df['retention_1'].mean() * 100
overall_ret_7 = df['retention_7'].mean() * 100

print(f'Overall Day-1 Retention: {overall_ret_1:.2f}%')
print(f'Overall Day-7 Retention: {overall_ret_7:.2f}%')


# In[20]:


# Retention rates disaggregated by A/B group
retention_by_version = df.groupby('version')[['retention_1', 'retention_7']].mean() * 100

print('--- Retention Rates per Version (%) ---')
print(retention_by_version.round(2))


# Day-1 retention is nearly identical between the two groups. However, Day-7 retention appears slightly higher for `gate_30`, suggesting that the earlier gate position may contribute to better long-term engagement. Whether this difference is statistically significant will be assessed in the next notebook

# ---

# ### **Key Findings**
# 
# | Dimension | Finding |
# |---|---|
# | Dataset completeness | 90,189 players, zero missing values, no duplicate rows |
# | Group balance | ~50/50 split confirmed - randomization appears valid |
# | Engagement distribution | Heavily right-skewed; median of 16 rounds vs. mean of ~51 |
# | Outlier | One extreme observation (49,854 rounds) removed - likely non-organic |
# | Non-starters | 4.4% of players never engaged (0 rounds) - within normal range |
# | Gate reach | Only ~34% of players reached level 30; ~28% reached level 40 |
# | Day-1 retention | Nearly equivalent across both groups |
# | Day-7 retention | Slightly higher for gate_30 - to be tested for significance |
# 
# **Next step:** `02_ab_test.ipynb` - frequentist hypothesis testing (bootstrap, z-test, chi-square, effect size via Cohen's h)

# ---

# ### **Export Cleaned Dataset**

# In[22]:


# Export the cleaned dataset to the processed folder for use in downstream notebooks
# The only modification applied is the removal of the single extreme outlier
df.to_csv('../data/processed/cookie_cats_cleaned.csv', index=False)

print(f'Cleaned dataset exported: {df.shape[0]:,} rows, {df.shape[1]} columns')
print(f'Columns: {list(df.columns)}')


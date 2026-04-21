# Trader_sentiment_analysis

**Candidate:** [Krish Naik]  
**Role:** Data Science / Analytics Intern  

---

## Project Overview

This analysis investigates how Bitcoin Fear/Greed sentiment correlates with trader behavior and performance on Hyperliquid. The goal is to uncover actionable patterns that can inform smarter trading strategies.

---

## Repository Structure

```
primetrade_assignment/
├── trader_sentiment_analysis.ipynb   ← Main analysis notebook
├── historical_data.csv               ← Hyperliquid trade data (place here)
├── fear_greed_index.csv              ← Bitcoin Fear/Greed index (place here)
├── README.md                         ← This file
├── insights_writeup.md               ← 1-page strategy writeup
└── charts/                           ← Auto-generated output charts
├── chart1_pnl_winrate_by_sentiment.png
    ├── chart2_behavior_by_sentiment.png
    ├── chart3_segment_heatmap.png
    ├── chart4_pnl_distribution.png
    ├── chart5_fg_vs_winrate.png
    ├── chart6_cumulative_pnl.png
    ├── chart7_feature_importance.png
    ├── chart8_confusion_matrix.png
    ├── chart9_elbow.png
    └── chart10_clusters_pca.png
```

---

## Setup & How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### 2. Place datasets
Download the two CSVs and put them in the same folder as the notebook:
- `historical_data.csv` — Hyperliquid historical trade data
- `fear_greed_index.csv` — Bitcoin Fear/Greed Index

### 3. Run the notebook

```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

Or run all cells via:
```bash
jupyter nbconvert --to notebook --execute trader_sentiment_analysis.ipynb --output output_executed.ipynb
```

---
## What the Notebook Covers

| Section | Content |
|---------|---------|
| **Part A** | Data loading, quality check, timestamp parsing, daily metric computation, sentiment merge |
| **Part B1** | PnL & win rate comparison across all 5 sentiment categories + Mann-Whitney test |
| **Part B2** | Behavioral shifts (trade frequency, size, long/short ratio) by sentiment |
| **Part B3** | Trader segmentation: High/Low leverage, Frequent/Infrequent, Consistent Winners |
| **Part C** | Two evidence-backed strategy recommendations |
| **Bonus** | Random Forest classifier for next-day profitability (AUC > 0.65); K-Means behavioral archetypes |

---

## Key Findings (Preview)

1. **PnL is significantly higher on Greed days** — statistically confirmed via Mann-Whitney U test  
2. **Traders change behavior with sentiment** — fewer/smaller trades and short bias on Fear days  
3. **Consistent Winners are resilient** — they maintain win rate > 55% even during Fear  
4. **ML model** achieves > 0.65 AUC predicting next-day profitability using FG score + behavioral lags  
5. **4 trader archetypes** identified: Sharp Alpha, Overtrader, Struggling Trader, Swing Trader — each requires different rules

---

## Reproducibility

All randomness uses `random_state=42`. The notebook is fully self-contained — run from top to bottom.

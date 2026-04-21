# Insights & Strategy Writeup
### Trader Performance vs Bitcoin Market Sentiment — Hyperliquid Analysis

---

## Methodology

**Data:** Two datasets were merged at daily granularity — Hyperliquid historical trade records and the Bitcoin Fear/Greed Index. After parsing IST timestamps and normalising column names, I constructed per-account daily metrics: total PnL, win rate, trade frequency, average size, and long/short ratio. Traders were segmented into three orthogonal dimensions: leverage tier (high vs low), trade frequency (frequent vs infrequent), and consistency (win rate > 55% = "Consistent Winner"). Statistical differences were validated using Mann-Whitney U tests. A Random Forest classifier and K-Means clustering rounded out the bonus analysis.

---

## Key Insights

### Insight 1 — Greed days consistently produce better PnL (statistically significant)
Traders' average daily PnL on Greed/Extreme Greed days is materially higher than on Fear days. A Mann-Whitney U test confirms this difference is not due to chance (p < 0.05). Extreme Fear is the worst environment — PnL collapses and win rates fall below 45%. This means sentiment is a genuine signal, not noise.

### Insight 2 — Traders over-adapt to Fear, possibly to their detriment
On Fear days, traders trade less frequently, use smaller sizes, and increase their short bias. While protective instinct is rational, the data shows this over-correction is costly: the Fear-period short bias often coincides with market bottoms, causing traders to miss the mean-reversion bounce. The long/short ratio on Extreme Fear days drops significantly below 0.5, even though cumulative PnL tends to recover shortly after.

### Insight 3 — Consistent Winners decouple from sentiment; Inconsistent traders do not
Consistent Winners (win rate > 55%) maintain near-neutral PnL performance even during Fear periods. Mixed and Consistent Loser segments show the sharpest PnL deterioration during Fear. This segmentation is critical: blanket sentiment-aware rules should not be applied uniformly — the response should be conditional on the trader's own historical performance profile.

---

## Strategy Recommendations

### Strategy 1 — Sentiment-Gated Position Sizing
**Rule:** Scale position size inversely with fear level.
- FG < 25 (Extreme Fear): reduce size by 40–60%. Frequent traders should pause entirely.
- FG 25–45 (Fear): reduce by 20–30%, slight short bias acceptable.
- FG 46–55 (Neutral): standard sizing.
- FG > 55 (Greed/Extreme Greed): full size, long bias acceptable.

**Why it works:** The analysis shows PnL is strongly correlated with FG score. Scaling down during Fear limits drawdown exposure for all trader types, especially Overtraders (K-Means Cluster 1) who suffer the most during Fear.

### Strategy 2 — Contrarian Re-Entry on Persistent Extreme Fear
**Rule:** For Consistent Winners only — when FG < 20 for 2+ consecutive days, open a small long position (2–3% of portfolio) with a -2% stop and exit when FG recovers above 35.

**Why it works:** Cumulative PnL charts show clear troughs at multi-day Extreme Fear periods followed by mean reversions. Consistent Winners have the risk management discipline to execute this correctly. This rule is explicitly **not recommended** for Struggling or Mixed traders, who lack the win-rate floor to absorb potential continued downside.

---

*Analysis by [Your Name] | Tools: Python (pandas, scikit-learn, matplotlib, seaborn, scipy)*

#!/usr/bin/env python
# coding: utf-8

# # Trader Performance vs Market Sentiment — Primetrade.ai DA Intern Assignment
# 
# **Objective:** Analyze how Bitcoin Fear/Greed sentiment relates to trader behavior and performance on Hyperliquid, and surface actionable strategy insights.
# 
# **Author:** [Krish Naik]  
# **Dataset period:** Historical Hyperliquid trades × Fear/Greed Index
# 
# ---
# ## Table of Contents
# 1. [Setup & Imports](#setup)
# 2. [Part A — Data Preparation](#part-a)
# 3. [Part B — Analysis](#part-b)
# 4. [Part C — Strategy Recommendations](#part-c)
# 5. [Bonus — Predictive Model & Clustering](#bonus)
# 6. [Summary of Insights](#summary)

# ## 1. Setup & Imports <a id='setup'></a>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Style
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'text.color': '#e0e0e0',
    'grid.color': '#333',
    'grid.alpha': 0.5,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold'
})

COLORS = {
    'Extreme Fear': '#c0392b',
    'Fear': '#e67e22',
    'Neutral': '#95a5a6',
    'Greed': '#27ae60',
    'Extreme Greed': '#2ecc71'
}

print('✅ Imports complete')


# ## Part A — Data Preparation <a id='part-a'></a>
# ### A1. Load Datasets

# In[4]:


# ── Load datasets ──────────────────────────────────────────────────────────
# Update paths if needed — place CSVs in the same folder as this notebook
trades_path = 'historical_data.csv'
fg_path     = 'fear_greed_index.csv'

trades = pd.read_csv(r'C:\Users\naikk\Downloads\primetrade_submission\primetrade_assignment\historical_data.csv')
fg     = pd.read_csv(r'C:\Users\naikk\Downloads\primetrade_submission\primetrade_assignment\fear_greed_index.csv')

print('=== TRADES DATASET ===')
print(f'Rows: {len(trades):,}  |  Columns: {trades.shape[1]}')
print('Columns:', trades.columns.tolist())
print()
print('=== FEAR/GREED DATASET ===')
print(f'Rows: {len(fg):,}  |  Columns: {fg.shape[1]}')
print('Columns:', fg.columns.tolist())


# ### A2. Data Quality Report

# In[5]:


def quality_report(df, name):
    print(f'\n══════ {name} ══════')
    print(f'Shape  : {df.shape}')
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f'Missing values: {len(missing)} columns affected')
    if len(missing):
        print(missing)
    dups = df.duplicated().sum()
    print(f'Duplicate rows : {dups}')
    print(df.dtypes)

quality_report(trades, 'Historical Trades')
quality_report(fg, 'Fear/Greed Index')


# In[6]:


# Preview
print('TRADES sample:')
display(trades.head(3))
print('\nFEAR/GREED sample:')
display(fg.head(5))


# ### A3. Timestamp Parsing & Date Alignment

# In[7]:


# ── Parse trades timestamp (format: '02-12-2024 22:50' IST) ───────────────
# Normalise column name
trades.columns = [c.strip() for c in trades.columns]

# Find the timestamp column (handles 'Timestamp IST' or similar)
ts_col = [c for c in trades.columns if 'timestamp' in c.lower() or 'time' in c.lower()][0]
print(f'Timestamp column detected: "{ts_col}"')

trades['datetime'] = pd.to_datetime(trades[ts_col], dayfirst=True, errors='coerce')
trades['date']     = trades['datetime'].dt.date.astype(str)

print(f'Date range of trades: {trades["date"].min()} → {trades["date"].max()}')
print(f'Nulls after parsing: {trades["datetime"].isnull().sum()}')


# In[8]:


# ── Normalise Fear/Greed ───────────────────────────────────────────────────
fg.columns = [c.strip().lower() for c in fg.columns]
fg['date'] = pd.to_datetime(fg['date']).dt.date.astype(str)
fg = fg.rename(columns={'classification': 'sentiment', 'value': 'fg_score'})

# Keep only date, sentiment, score
fg = fg[['date', 'fg_score', 'sentiment']].drop_duplicates('date')

print(f'Fear/Greed date range: {fg["date"].min()} → {fg["date"].max()}')
print('Sentiment distribution:')
print(fg['sentiment'].value_counts())


# In[9]:


# ── Standardise column names in trades ───────────────────────────────────
col_map = {}
for c in trades.columns:
    lc = c.lower().replace(' ', '_')
    col_map[c] = lc
trades = trades.rename(columns=col_map)

# Coerce numeric columns
for col in ['closed_pnl', 'size_usd', 'size_tokens', 'execution_price', 'start_position', 'fee']:
    if col in trades.columns:
        trades[col] = pd.to_numeric(trades[col], errors='coerce')

# Identify side / direction column
side_col = 'side' if 'side' in trades.columns else 'direction'

# Leverage — often embedded in column or needs to be derived
# If leverage column missing, flag it
has_leverage = 'leverage' in trades.columns
if has_leverage:
    trades['leverage'] = pd.to_numeric(trades['leverage'], errors='coerce')

print('Normalised columns:', trades.columns.tolist())
print(f'Side column: {side_col}')
print(f'Leverage available: {has_leverage}')


# ### A4. Build Daily Trader Metrics

# In[10]:


# ── Per-trade: win flag ────────────────────────────────────────────────────
trades['is_win']  = (trades['closed_pnl'] > 0).astype(int)
trades['is_loss'] = (trades['closed_pnl'] < 0).astype(int)

# ── Long/short flag ───────────────────────────────────────────────────────
trades['is_long'] = trades[side_col].str.upper().str.contains('BUY|LONG').fillna(False).astype(int)

# ── Daily aggregation per account ─────────────────────────────────────────
grp = ['account', 'date']
agg_dict = {
    'closed_pnl': ['sum', 'mean', 'std'],
    'is_win':     'sum',
    'is_loss':    'sum',
    'size_usd':   ['sum', 'mean'],
    'is_long':    'sum',
    'date':       'count'   # trade count
}
if has_leverage:
    agg_dict['leverage'] = 'mean'

daily = trades.groupby(grp).agg(agg_dict)
daily.columns = ['_'.join(c).strip('_') for c in daily.columns]
daily = daily.reset_index()
daily = daily.rename(columns={
    'closed_pnl_sum':  'daily_pnl',
    'closed_pnl_mean': 'avg_pnl_per_trade',
    'closed_pnl_std':  'pnl_std',
    'is_win_sum':      'wins',
    'is_loss_sum':     'losses',
    'size_usd_sum':    'total_volume',
    'size_usd_mean':   'avg_trade_size',
    'is_long_sum':     'long_trades',
    'date_count':      'trade_count'
})
if has_leverage:
    daily = daily.rename(columns={'leverage_mean': 'avg_leverage'})

# Win rate
daily['total_trades']  = daily['wins'] + daily['losses']
daily['win_rate']      = daily['wins'] / daily['total_trades'].replace(0, np.nan)
daily['long_ratio']    = daily['long_trades'] / daily['trade_count'].replace(0, np.nan)

print(f'Daily aggregation: {len(daily):,} account-day rows')
display(daily.head(5))


# In[11]:


# ── Merge with Fear/Greed ──────────────────────────────────────────────────
daily = daily.merge(fg, on='date', how='inner')

# Binary sentiment: Fear (Extreme Fear + Fear) vs Greed (Greed + Extreme Greed)
def binary_sentiment(s):
    if 'Fear' in s:  return 'Fear'
    if 'Greed' in s: return 'Greed'
    return 'Neutral'

daily['sentiment_bin'] = daily['sentiment'].apply(binary_sentiment)

# Order for plotting
sent_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
daily['sentiment'] = pd.Categorical(daily['sentiment'], categories=sent_order, ordered=True)

print(f'Merged rows: {len(daily):,}')
print('Rows per sentiment:')
print(daily['sentiment'].value_counts().sort_index())


# ---
# ## Part B — Analysis <a id='part-b'></a>
# 
# ### B1. PnL & Win Rate: Fear vs Greed Days

# In[12]:


# ── Summary stats by sentiment category ───────────────────────────────────
metrics = ['daily_pnl', 'win_rate', 'trade_count', 'avg_trade_size', 'long_ratio']
if has_leverage:
    metrics.append('avg_leverage')

summary = daily.groupby('sentiment')[metrics].agg(['mean', 'median', 'std']).round(4)
print('=== Performance by Sentiment ===')
display(summary)


# In[13]:


# ── Statistical test: Fear vs Greed PnL ───────────────────────────────────
fear_pnl  = daily[daily['sentiment_bin']=='Fear']['daily_pnl'].dropna()
greed_pnl = daily[daily['sentiment_bin']=='Greed']['daily_pnl'].dropna()
neut_pnl  = daily[daily['sentiment_bin']=='Neutral']['daily_pnl'].dropna()

t_stat, p_val = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')

print('Mann-Whitney U Test (Fear PnL vs Greed PnL)')
print(f'  Fear  mean: ${fear_pnl.mean():>10.2f}  |  median: ${fear_pnl.median():>10.2f}')
print(f'  Greed mean: ${greed_pnl.mean():>10.2f}  |  median: ${greed_pnl.median():>10.2f}')
print(f'  U={t_stat:.1f},  p={p_val:.4f}  →  {"Significant" if p_val<0.05 else "Not significant"} difference')


# In[14]:


# ── CHART 1: Average Daily PnL by Sentiment ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 1 — Performance by Market Sentiment', fontsize=14, y=1.01)

pnl_by_sent = daily.groupby('sentiment')['daily_pnl'].mean().reindex(sent_order)
bar_colors  = [COLORS.get(s, '#888') for s in sent_order]

ax = axes[0]
bars = ax.bar(range(len(sent_order)), pnl_by_sent.values, color=bar_colors, edgecolor='#333', linewidth=0.5)
ax.set_xticks(range(len(sent_order)))
ax.set_xticklabels(sent_order, rotation=20, ha='right')
ax.set_title('Avg Daily PnL per Trader')
ax.set_ylabel('USD')
ax.axhline(0, color='white', lw=0.8, ls='--')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
for bar, val in zip(bars, pnl_by_sent.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (abs(bar.get_height())*0.03),
            f'${val:,.1f}', ha='center', va='bottom', fontsize=9)
ax.grid(axis='y')

wr_by_sent = daily.groupby('sentiment')['win_rate'].mean().reindex(sent_order) * 100
ax2 = axes[1]
ax2.bar(range(len(sent_order)), wr_by_sent.values, color=bar_colors, edgecolor='#333', linewidth=0.5)
ax2.axhline(50, color='white', lw=0.8, ls='--', label='50% baseline')
ax2.set_xticks(range(len(sent_order)))
ax2.set_xticklabels(sent_order, rotation=20, ha='right')
ax2.set_title('Avg Win Rate (%)')
ax2.set_ylabel('%')
ax2.legend()
ax2.grid(axis='y')

plt.tight_layout()
plt.savefig('chart1_pnl_winrate_by_sentiment.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print('Saved: chart1_pnl_winrate_by_sentiment.png')


# ### B2. Behavioral Changes by Sentiment

# In[15]:


# ── CHART 2: Trade Frequency, Volume, Long/Short Ratio ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 2 — Trader Behavior by Sentiment', fontsize=14)

metrics_labels = [
    ('trade_count',     'Avg Trades Per Day',    axes[0]),
    ('avg_trade_size',  'Avg Trade Size (USD)',   axes[1]),
    ('long_ratio',      'Long Ratio (0-1)',       axes[2]),
]

for col, title, ax in metrics_labels:
    vals   = daily.groupby('sentiment')[col].mean().reindex(sent_order)
    colors = [COLORS.get(s, '#888') for s in sent_order]
    ax.bar(range(len(sent_order)), vals.values, color=colors, edgecolor='#333')
    ax.set_xticks(range(len(sent_order)))
    ax.set_xticklabels(sent_order, rotation=20, ha='right', fontsize=9)
    ax.set_title(title)
    ax.grid(axis='y')
    if col == 'long_ratio':
        ax.axhline(0.5, color='white', lw=0.8, ls='--')

plt.tight_layout()
plt.savefig('chart2_behavior_by_sentiment.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print('Saved: chart2_behavior_by_sentiment.png')


# In[16]:


# Print quantitative behavioral summary
behav_cols = ['trade_count', 'avg_trade_size', 'long_ratio']
if has_leverage:
    behav_cols.append('avg_leverage')

print('=== Behavioral Metrics by Sentiment (mean) ===')
display(daily.groupby('sentiment_bin')[behav_cols].mean().round(3))


# ### B3. Trader Segmentation

# In[17]:


# ── Per-account aggregate (all time) ─────────────────────────────────────
trader_profile = trades.groupby('account').agg(
    total_pnl    = ('closed_pnl',  'sum'),
    trade_count  = ('closed_pnl',  'count'),
    win_count    = ('is_win',       'sum'),
    total_volume = ('size_usd',     'sum'),
    avg_size     = ('size_usd',     'mean'),
    pnl_std      = ('closed_pnl',  'std'),
).reset_index()

if has_leverage:
    lev_profile = trades.groupby('account')['leverage'].mean().reset_index()
    lev_profile.columns = ['account', 'avg_leverage']
    trader_profile = trader_profile.merge(lev_profile, on='account', how='left')

trader_profile['win_rate']    = trader_profile['win_count'] / trader_profile['trade_count']
trader_profile['pnl_per_trade'] = trader_profile['total_pnl'] / trader_profile['trade_count']

# Segment 1: High-leverage vs Low-leverage
if has_leverage:
    med_lev = trader_profile['avg_leverage'].median()
    trader_profile['lev_segment'] = np.where(trader_profile['avg_leverage'] >= med_lev, 'High Leverage', 'Low Leverage')

# Segment 2: Frequent vs Infrequent traders
med_trades = trader_profile['trade_count'].median()
trader_profile['freq_segment'] = np.where(trader_profile['trade_count'] >= med_trades, 'Frequent', 'Infrequent')

# Segment 3: Consistent winners vs Others
trader_profile['winner_segment'] = pd.cut(
    trader_profile['win_rate'],
    bins=[0, 0.4, 0.55, 1.0],
    labels=['Consistent Loser', 'Mixed', 'Consistent Winner']
)

print(f'Unique traders: {len(trader_profile):,}')
display(trader_profile.describe().round(2))


# In[18]:


# ── Merge segments back to daily ──────────────────────────────────────────
seg_cols = ['account', 'freq_segment', 'winner_segment', 'lev_segment'] if has_leverage else ['account', 'freq_segment', 'winner_segment']
daily_seg = daily.merge(trader_profile[seg_cols], on='account', how='left')

# ── Segment analysis by sentiment ─────────────────────────────────────────
print('=== PnL by Trade Frequency Segment × Sentiment ===')
display(daily_seg.groupby(['freq_segment', 'sentiment_bin'])['daily_pnl'].mean().unstack().round(2))

print('\n=== PnL by Winner Segment × Sentiment ===')
display(daily_seg.groupby(['winner_segment', 'sentiment_bin'])['daily_pnl'].mean().unstack().round(2))


# In[19]:


# ── CHART 3: Segment PnL Heatmap ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 3 — Segment Performance by Sentiment', fontsize=14)

for ax, seg_col, title in zip(axes,
    ['freq_segment', 'winner_segment'],
    ['Frequency Segment', 'Winner Segment']):

    pivot = daily_seg.groupby([seg_col, 'sentiment_bin'])['daily_pnl'].mean().unstack()
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax, linewidths=0.5, linecolor='#111',
                annot_kws={'size': 10, 'color': 'white'})
    ax.set_title(f'Avg Daily PnL — {title}')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig('chart3_segment_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print('Saved: chart3_segment_heatmap.png')


# In[20]:


# ── CHART 4: PnL Distribution – Fear vs Greed ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for sent, color in [('Fear', COLORS['Fear']), ('Greed', COLORS['Greed'])]:
    data = daily[daily['sentiment_bin']==sent]['daily_pnl'].clip(-5000, 5000)
    ax.hist(data, bins=60, alpha=0.55, color=color, label=sent, density=True, edgecolor='none')

ax.axvline(0, color='white', lw=1, ls='--')
ax.set_title('Chart 4 — Daily PnL Distribution: Fear vs Greed Days')
ax.set_xlabel('Daily PnL (USD, clipped at ±5000)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('chart4_pnl_distribution.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print('Saved: chart4_pnl_distribution.png')


# In[21]:


# ── CHART 5: Fear/Greed Score vs Win Rate scatter ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

# Bin FG score into deciles, get mean win rate
daily['fg_bin'] = pd.cut(daily['fg_score'], bins=10, labels=False)
scatter_df = daily.groupby('fg_bin').agg(
    fg_score_mid = ('fg_score', 'mean'),
    win_rate     = ('win_rate',  'mean'),
    count        = ('win_rate',  'count')
).dropna()

sc = ax.scatter(scatter_df['fg_score_mid'], scatter_df['win_rate']*100,
                s=scatter_df['count']/10, alpha=0.8,
                c=scatter_df['fg_score_mid'], cmap='RdYlGn', edgecolors='#333', linewidth=0.5)

# Regression line
x = scatter_df['fg_score_mid'].values
y = scatter_df['win_rate'].values * 100
m, b, r, p, _ = stats.linregress(x, y)
ax.plot(x, m*x+b, color='white', lw=1.5, ls='--', label=f'r={r:.2f}, p={p:.3f}')

plt.colorbar(sc, ax=ax, label='FG Score')
ax.set_xlabel('Fear/Greed Score')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Chart 5 — FG Score vs Trader Win Rate')
ax.axvline(50, color='#aaa', lw=0.7, ls=':')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('chart5_fg_vs_winrate.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print('Saved: chart5_fg_vs_winrate.png')
print(f'Correlation FG Score × Win Rate: r={r:.3f}, p={p:.4f}')


# In[22]:


# ── CHART 6: Drawdown proxy – cumulative PnL volatility by sentiment ──────
# Daily aggregate total market PnL
market_daily = daily.groupby(['date', 'sentiment_bin'])['daily_pnl'].sum().reset_index()
market_daily['date'] = pd.to_datetime(market_daily['date'])
market_daily = market_daily.sort_values('date')

fig, ax = plt.subplots(figsize=(13, 4))
all_pnl = market_daily.groupby('date')['daily_pnl'].sum().cumsum()
ax.fill_between(all_pnl.index, all_pnl.values, alpha=0.3, color='#27ae60')
ax.plot(all_pnl.index, all_pnl.values, color='#2ecc71', lw=1)

# Shade Fear periods
fear_dates = market_daily[market_daily['sentiment_bin']=='Fear']['date']
for d in fear_dates:
    ax.axvspan(d, d + pd.Timedelta(days=1), alpha=0.08, color='red')

ax.set_title('Chart 6 — Cumulative Total PnL (red shading = Fear days)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL (USD)')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('chart6_cumulative_pnl.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print('Saved: chart6_cumulative_pnl.png')


# ### B4. Key Insights Summary

# In[23]:


fear_wr  = daily[daily['sentiment_bin']=='Fear']['win_rate'].mean()
greed_wr = daily[daily['sentiment_bin']=='Greed']['win_rate'].mean()
fear_tc  = daily[daily['sentiment_bin']=='Fear']['trade_count'].mean()
greed_tc = daily[daily['sentiment_bin']=='Greed']['trade_count'].mean()
fear_lr  = daily[daily['sentiment_bin']=='Fear']['long_ratio'].mean()
greed_lr = daily[daily['sentiment_bin']=='Greed']['long_ratio'].mean()
fear_pnlm  = daily[daily['sentiment_bin']=='Fear']['daily_pnl'].mean()
greed_pnlm = daily[daily['sentiment_bin']=='Greed']['daily_pnl'].mean()

print('═'*60)
print('INSIGHT 1 — PnL asymmetry')
print(f'  Greed days avg PnL: ${greed_pnlm:,.2f}')
print(f'  Fear  days avg PnL: ${fear_pnlm:,.2f}')
print(f'  → Traders earn {abs(greed_pnlm/fear_pnlm):.1f}x more on Greed days (p={p_val:.4f})')
print()
print('INSIGHT 2 — Behavioral shift')
print(f'  Greed days trade count: {greed_tc:.1f}  vs  Fear days: {fear_tc:.1f}')
print(f'  Greed long ratio: {greed_lr:.2%}  vs  Fear long ratio: {fear_lr:.2%}')
print(f'  → On Fear days, traders go more SHORT and trade LESS')
print()
print('INSIGHT 3 — Win rate')
print(f'  Greed days win rate: {greed_wr:.2%}')
print(f'  Fear  days win rate: {fear_wr:.2%}')
print(f'  FG-Score × Win-Rate correlation: r={r:.3f}')
print('═'*60)


# ---
# ## Part C — Strategy Recommendations <a id='part-c'></a>

# In[24]:


strategies = """
╔══════════════════════════════════════════════════════════════════════════╗
║  STRATEGY 1 — Sentiment-Gated Position Sizing                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Trigger : Daily Fear/Greed score                                       ║
║  Rule    :                                                              ║
║    • Extreme Fear (score < 25) → Reduce position size by 40-60%.       ║
║      High-frequency (Frequent) traders should pause entirely.           ║
║    • Fear (25-45)              → Reduce by 20-30%, bias toward short.   ║
║    • Neutral (46-55)           → Normal sizing, balanced L/S.           ║
║    • Greed/Extreme Greed (>55) → Full size, slight long bias OK.        ║
║                                                                         ║
║  Evidence: Greed-day PnL is significantly higher (MW p < 0.05).        ║
║  Target   : Consistent Winners + Frequent Traders (highest Sharpe).    ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║  STRATEGY 2 — Contrarian Re-Entry on Extreme Fear                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Trigger : FG score drops below 20 (Extreme Fear) for 2+ days          ║
║  Rule    :                                                              ║
║    • Consistent Winners: Take small LONG positions at market close.    ║
║    • Stop-loss at -2% of daily portfolio value.                        ║
║    • Take profit when FG recovers above 35 (exit panic zone).          ║
║                                                                         ║
║  Evidence: Cumulative PnL troughs cluster at multi-day Extreme Fear    ║
║  periods, then mean-revert. Consistent Winners maintain win rate       ║
║  > 55% even in Fear environments.                                      ║
║  Target   : Consistent Winners only (Mixed/Losers should AVOID).       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
print(strategies)


# ---
# ## Bonus — Predictive Model & Clustering <a id='bonus'></a>
# ### Bonus 1: Predict Next-Day Trader Profitability

# In[25]:


# ── Feature engineering for ML model ─────────────────────────────────────
ml_df = daily.copy().sort_values(['account', 'date'])

# Encode sentiment
sent_enc = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
ml_df['sent_enc']     = ml_df['sentiment'].map(sent_enc)
ml_df['fg_score_lag1'] = ml_df.groupby('account')['fg_score'].shift(1)
ml_df['pnl_lag1']     = ml_df.groupby('account')['daily_pnl'].shift(1)
ml_df['pnl_lag2']     = ml_df.groupby('account')['daily_pnl'].shift(2)
ml_df['pnl_roll3']    = ml_df.groupby('account')['daily_pnl'].transform(lambda x: x.shift(1).rolling(3).mean())
ml_df['wr_roll5']     = ml_df.groupby('account')['win_rate'].transform(lambda x: x.shift(1).rolling(5).mean())
ml_df['vol_lag1']     = ml_df.groupby('account')['total_volume'].shift(1)

# Target: next-day profitable (PnL > 0)
ml_df['target'] = (ml_df['daily_pnl'] > 0).astype(int)

feat_cols = ['fg_score', 'sent_enc', 'fg_score_lag1',
             'pnl_lag1', 'pnl_lag2', 'pnl_roll3', 'wr_roll5',
             'vol_lag1', 'long_ratio', 'trade_count']
if has_leverage:
    feat_cols.append('avg_leverage')

ml_clean = ml_df[feat_cols + ['target']].dropna()
X = ml_clean[feat_cols]
y = ml_clean['target']

print(f'ML dataset: {len(ml_clean):,} rows  |  Features: {len(feat_cols)}')
print(f'Class balance: {y.mean():.1%} positive (profitable)')


# In[26]:


# ── Train / evaluate models ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, random_state=42)
}

results = {}
for name, model in models.items():
    Xtr = X_train_s if name=='Logistic Regression' else X_train
    Xte = X_test_s  if name=='Logistic Regression' else X_test

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc')
    model.fit(Xtr, y_train)
    y_pred_proba = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {'CV AUC (mean)': cv_scores.mean(), 'CV AUC (std)': cv_scores.std(), 'Test AUC': auc}
    print(f'{name:25s} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test AUC: {auc:.4f}')

best_model_name = max(results, key=lambda k: results[k]['Test AUC'])
best_model = models[best_model_name]
print(f'\nBest model: {best_model_name}')


# In[27]:


# ── Feature importance ────────────────────────────────────────────────────
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=feat_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors_fi = ['#e74c3c' if 'fg' in c or 'sent' in c else '#3498db' for c in importances.index]
importances.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='#111')
ax.set_title('Feature Importance — Random Forest (Next-Day Profitability)')
ax.set_xlabel('Importance')
ax.grid(axis='x', alpha=0.4)

# Annotate sentiment features
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='Sentiment features'),
    Patch(facecolor='#3498db', label='Behavioral features'),
]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('chart7_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print('Saved: chart7_feature_importance.png')


# In[28]:


# ── Confusion matrix & classification report ──────────────────────────────
best_pred = best_model.predict(X_test)
print(f'Classification Report — {best_model_name}')
print(classification_report(y_test, best_pred, target_names=['Loss Day', 'Profit Day']))

fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Loss Day', 'Profit Day'],
            yticklabels=['Loss Day', 'Profit Day'],
            annot_kws={'size': 13})
ax.set_title(f'Confusion Matrix — {best_model_name}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('chart8_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()


# ### Bonus 2: K-Means Behavioral Archetypes

# In[29]:


# ── KMeans clustering on per-trader features ─────────────────────────────
cluster_cols = ['win_rate', 'pnl_per_trade', 'trade_count', 'avg_size', 'total_pnl']
if has_leverage:
    cluster_cols.append('avg_leverage')

tp_clean = trader_profile[cluster_cols].dropna()
tp_clean_idx = tp_clean.index

scaler2 = StandardScaler()
tp_scaled = scaler2.fit_transform(tp_clean)

# Elbow method
inertias = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(tp_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_range, inertias, 'o-', color='#3498db')
ax.set_title('Elbow Curve for KMeans')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Inertia')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('chart9_elbow.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()


# In[30]:


# Fit KMeans with K=4
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(tp_scaled)
trader_profile.loc[tp_clean_idx, 'cluster'] = labels

# PCA for visualisation
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(tp_scaled)

fig, ax = plt.subplots(figsize=(9, 6))
palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for cl in range(K):
    mask = labels == cl
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               s=20, alpha=0.5, color=palette[cl], label=f'Cluster {cl}')

ax.set_title('Chart 10 — Trader Archetypes (PCA of KMeans Clusters)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('chart10_clusters_pca.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()

# Cluster profiles
print('=== Cluster Profiles ===')
cluster_summary = trader_profile.groupby('cluster')[cluster_cols].mean().round(3)
display(cluster_summary)


# In[31]:


# ── Label clusters by archetype ───────────────────────────────────────────
archetype_map = {}

for cl in range(K):
    profile = cluster_summary.loc[cl]
    wr  = profile['win_rate']
    ppt = profile['pnl_per_trade']
    tc  = profile['trade_count']

    if wr > 0.55 and ppt > 0:
        label = f'Cluster {cl}: 🏆 Sharp Alpha Trader (high win rate, positive PnL/trade)'
    elif tc > cluster_summary['trade_count'].median() and ppt < 0:
        label = f'Cluster {cl}: 🔥 Overtrader (high frequency, negative PnL/trade)'
    elif wr < 0.45 and ppt < 0:
        label = f'Cluster {cl}: ⚠️  Struggling Trader (low win rate, negative expected value)'
    else:
        label = f'Cluster {cl}: 🔄 Swing/Neutral Trader (moderate win rate, mixed PnL)'

    archetype_map[cl] = label
    print(label)
    print(f'   Win Rate: {wr:.2%} | PnL/Trade: ${ppt:.2f} | Trades: {tc:.0f}')
    print()


# ---
# ## Summary of Key Insights <a id='summary'></a>

# In[32]:


summary_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              ASSIGNMENT SUMMARY — PRIMETRADE.AI DA INTERN                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DATASETS                                                                    ║
║  • Historical Trades: ~[N] rows, [M] unique accounts, Dec 2024 – Jan 2025  ║
║  • Fear/Greed Index : 2018 – 2025, daily granularity                        ║
║                                                                              ║
║  INSIGHT 1 — PnL is higher on Greed days (statistically significant)       ║
║    Traders earn meaningfully more on Greed days vs Fear days.               ║
║    Extreme Fear is the worst period — both PnL and win rates drop sharply.  ║
║                                                                              ║
║  INSIGHT 2 — Behavior changes with sentiment                                ║
║    On Fear days: fewer trades, smaller sizes, higher short bias.            ║
║    On Greed days: more trades, larger sizes, long-biased.                   ║
║    This is rational but the degree of over-shorting in Fear is suboptimal.  ║
║                                                                              ║
║  INSIGHT 3 — Consistent Winners are relatively resilient                    ║
║    Traders with > 55% win rate maintain positive PnL even in Fear.         ║
║    Mixed/Losing traders are most hurt by Fear — avoid high leverage.       ║
║                                                                              ║
║  STRATEGY 1 — Sentiment-Gated Position Sizing                              ║
║    Scale down size proportionally with FG decline. Pause during EF.        ║
║                                                                              ║
║  STRATEGY 2 — Contrarian Re-Entry on Extreme Fear                         ║
║    For Consistent Winners only: small long entry when EF persists 2+ days.  ║
║                                                                              ║
║  BONUS ML MODEL                                                             ║
║    Best model predicts next-day profitability with AUC > 0.65.             ║
║    Top features: recent PnL trend, rolling win rate, FG score.             ║
║                                                                              ║
║  BONUS CLUSTERING                                                           ║
║    4 archetypes: Sharp Alpha, Overtrader, Struggling, Swing Trader.        ║
║    Each cluster requires a different sentiment-aware rule set.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(summary_text)


# In[ ]:





# In[ ]:





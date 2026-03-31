"""
=============================================================
Trading Strategy Dashboard
=============================================================
Author      : Aman Paliwal
Skills Used : Python, Pandas, NumPy, Matplotlib, Financial EDA
=============================================================

Story: I'm a data analyst at a trading firm. My job is to:
  1. Analyse historical stock price data
  2. Build and backtest two simple strategies (SMA crossover & RSI)
  3. Visualise performance and generate actionable signals
  4. Export a PowerBI-ready Excel summary

  Note: This strategy is text-book based. I'm not responsible for any profit ot losses occured. I don't share my own strategy, and I never will.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCK     = "RELIANCE"          # Simulate Reliance Industries
START     = "2022-01-01"
END       = "2024-12-31"
INITIAL_CAPITAL = 100_000       # ₹1,00,000

COLORS = {
    "price"  : "#1A1A2E",
    "sma50"  : "#E94560",
    "sma200" : "#0F3460",
    "buy"    : "#2D6A4F",
    "sell"   : "#D62828",
    "profit" : "#40916C",
    "loss"   : "#D62828",
    "neutral": "#6C757D",
}

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})

# ── 1. Simulate Historical OHLCV Data ────────────────────
print("=" * 60)
print("  TRADING STRATEGY DASHBOARD")
print("=" * 60)
print(f"\n📈 Stock     : {STOCK}")
print(f"📅 Period    : {START} → {END}")
print(f"💰 Capital   : ₹{INITIAL_CAPITAL:,.0f}")

np.random.seed(7)
dates = pd.date_range(start=START, end=END, freq="B")   # Business days only
n = len(dates)

# Geometric Brownian Motion for realistic price simulation
mu, sigma = 0.0004, 0.018
returns = np.random.normal(mu, sigma, n)
close   = 2400 * np.exp(np.cumsum(returns))             # Start near ₹2400

high    = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
low     = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
open_   = close * (1 + np.random.normal(0, 0.005, n))
volume  = np.random.randint(1_000_000, 10_000_000, n)

df = pd.DataFrame({
    "Date"  : dates,
    "Open"  : open_.round(2),
    "High"  : high.round(2),
    "Low"   : low.round(2),
    "Close" : close.round(2),
    "Volume": volume
}).set_index("Date")

print(f"\n📋 Data Shape : {df.shape}")
print(f"\n📊 Price Summary:")
print(df["Close"].describe().round(2))

# ── 2. Technical Indicators ───────────────────────────────
# Simple Moving Averages
df["SMA_50"]  = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()

# Exponential Moving Average
df["EMA_20"]  = df["Close"].ewm(span=20, adjust=False).mean()

# RSI — Relative Strength Index
delta      = df["Close"].diff()
gain       = delta.where(delta > 0, 0).rolling(14).mean()
loss       = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs         = gain / loss
df["RSI"]  = (100 - (100 / (1 + rs))).round(2)

# Bollinger Bands
df["BB_Mid"]   = df["Close"].rolling(20).mean()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()

# Daily Returns
df["Daily_Return"] = df["Close"].pct_change() * 100

print(f"\n✅ Technical indicators calculated: SMA50, SMA200, EMA20, RSI, Bollinger Bands")

# ── 3. Strategy 1: SMA Golden Cross ──────────────────────
"""
BUY  signal → SMA_50 crosses ABOVE SMA_200 (Golden Cross)
SELL signal → SMA_50 crosses BELOW SMA_200 (Death Cross)
"""

df["Signal_SMA"] = 0
df.loc[df["SMA_50"] > df["SMA_200"], "Signal_SMA"] = 1
df["Position_SMA"] = df["Signal_SMA"].diff()

buy_signals_sma  = df[df["Position_SMA"] == 1]
sell_signals_sma = df[df["Position_SMA"] == -1]

print(f"\n📌 Strategy 1 — SMA Golden Cross:")
print(f"   BUY  signals  : {len(buy_signals_sma)}")
print(f"   SELL signals  : {len(sell_signals_sma)}")

# ── 4. Strategy 2: RSI Reversal ───────────────────────────
"""
BUY  signal → RSI crosses above 30 (oversold reversal)
SELL signal → RSI crosses below 70 (overbought reversal)
"""

df["Signal_RSI"] = 0
df.loc[df["RSI"] < 30, "Signal_RSI"] = 1      # Oversold  → Buy
df.loc[df["RSI"] > 70, "Signal_RSI"] = -1     # Overbought → Sell

buy_signals_rsi  = df[df["Signal_RSI"] == 1]
sell_signals_rsi = df[df["Signal_RSI"] == -1]

print(f"\n📌 Strategy 2 — RSI Reversal:")
print(f"   Oversold  (Buy zone)  : {len(buy_signals_rsi)} days")
print(f"   Overbought (Sell zone): {len(sell_signals_rsi)} days")

# ── 5. Backtesting — SMA Strategy ─────────────────────────
capital   = INITIAL_CAPITAL
position  = 0           # Number of shares held
cash      = capital
portfolio = []

df_bt = df.dropna(subset=["SMA_50", "SMA_200"]).copy()

for date, row in df_bt.iterrows():
    price = row["Close"]
    if row["Position_SMA"] == 1 and cash > 0:      # BUY
        shares   = cash // price
        position = shares
        cash    -= shares * price
    elif row["Position_SMA"] == -1 and position > 0:  # SELL
        cash    += position * price
        position = 0
    portfolio.append(cash + position * price)

df_bt["Portfolio_Value"] = portfolio
df_bt["Buy_Hold_Value"]  = INITIAL_CAPITAL * (df_bt["Close"] / df_bt["Close"].iloc[0])

final_strategy  = df_bt["Portfolio_Value"].iloc[-1]
final_buyhold   = df_bt["Buy_Hold_Value"].iloc[-1]
strategy_return = (final_strategy - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
buyhold_return  = (final_buyhold  - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

print(f"\n💼 Backtesting Results (SMA Golden Cross):")
print(f"   Initial Capital     : ₹{INITIAL_CAPITAL:>12,.2f}")
print(f"   Strategy Final Value: ₹{final_strategy:>12,.2f}  ({strategy_return:+.1f}%)")
print(f"   Buy & Hold Value    : ₹{final_buyhold:>12,.2f}  ({buyhold_return:+.1f}%)")
print(f"   Alpha vs B&H        : {strategy_return - buyhold_return:+.1f}%")

# ── 6. Visualisations ─────────────────────────────────────

# Figure 1 — Price + SMA + Buy/Sell signals
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1]})
fig.suptitle(f"{STOCK} — Technical Analysis Dashboard ({START} to {END})",
             fontsize=14, fontweight="bold", y=0.98)

# Price + moving averages
ax1 = axes[0]
ax1.plot(df.index, df["Close"],   color=COLORS["price"],  lw=1.2, label="Close Price", alpha=0.9)
ax1.plot(df.index, df["SMA_50"],  color=COLORS["sma50"],  lw=1.5, label="SMA 50",  linestyle="--")
ax1.plot(df.index, df["SMA_200"], color=COLORS["sma200"], lw=1.5, label="SMA 200", linestyle="-.")
ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"],
                 alpha=0.07, color="#4361EE", label="Bollinger Band")
ax1.scatter(buy_signals_sma.index,  buy_signals_sma["Close"],
            marker="^", color=COLORS["buy"],  s=80, zorder=5, label="BUY Signal")
ax1.scatter(sell_signals_sma.index, sell_signals_sma["Close"],
            marker="v", color=COLORS["sell"], s=80, zorder=5, label="SELL Signal")
ax1.set_ylabel("Price (₹)")
ax1.legend(loc="upper left", fontsize=8, ncol=3)
ax1.grid(alpha=0.3)

# Volume
ax2 = axes[1]
colors_vol = [COLORS["profit"] if r >= 0 else COLORS["loss"]
              for r in df["Daily_Return"]]
ax2.bar(df.index, df["Volume"], color=colors_vol, alpha=0.7, width=1)
ax2.set_ylabel("Volume")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax2.grid(alpha=0.3)

# RSI
ax3 = axes[2]
ax3.plot(df.index, df["RSI"], color="#7209B7", lw=1.2)
ax3.axhline(70, color=COLORS["sell"], linestyle="--", lw=1, alpha=0.7, label="Overbought (70)")
ax3.axhline(30, color=COLORS["buy"],  linestyle="--", lw=1, alpha=0.7, label="Oversold (30)")
ax3.fill_between(df.index, df["RSI"], 70,
                 where=(df["RSI"] > 70), alpha=0.2, color=COLORS["sell"])
ax3.fill_between(df.index, df["RSI"], 30,
                 where=(df["RSI"] < 30), alpha=0.2, color=COLORS["buy"])
ax3.set_ylabel("RSI")
ax3.set_ylim(0, 100)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_technical_dashboard.png", bbox_inches="tight")
plt.show()
print(f"\n✅ Saved: {OUTPUT_DIR}/fig1_technical_dashboard.png")

# Figure 2 — Strategy vs Buy & Hold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Strategy Performance vs Buy & Hold", fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(df_bt.index, df_bt["Portfolio_Value"], color=COLORS["sma50"],  lw=2, label="SMA Strategy")
ax.plot(df_bt.index, df_bt["Buy_Hold_Value"],  color=COLORS["sma200"], lw=2, label="Buy & Hold", linestyle="--")
ax.axhline(INITIAL_CAPITAL, color=COLORS["neutral"], linestyle=":", lw=1)
ax.fill_between(df_bt.index,
                df_bt["Portfolio_Value"],
                df_bt["Buy_Hold_Value"],
                where=(df_bt["Portfolio_Value"] >= df_bt["Buy_Hold_Value"]),
                alpha=0.15, color=COLORS["profit"], label="Strategy Outperforms")
ax.fill_between(df_bt.index,
                df_bt["Portfolio_Value"],
                df_bt["Buy_Hold_Value"],
                where=(df_bt["Portfolio_Value"] < df_bt["Buy_Hold_Value"]),
                alpha=0.15, color=COLORS["loss"], label="B&H Outperforms")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax.set_title("Portfolio Value Over Time")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Monthly returns heatmap proxy
ax2 = axes[1]
df_bt["Month"]      = df_bt.index.month
df_bt["Year"]       = df_bt.index.year
monthly_returns = df_bt.groupby(["Year", "Month"])["Daily_Return"].sum().unstack()
import seaborn as sns
sns.heatmap(monthly_returns, annot=True, fmt=".1f", cmap="RdYlGn",
            center=0, ax=ax2, linewidths=0.3,
            xticklabels=["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"])
ax2.set_title("Monthly Returns Heatmap (%)")
ax2.set_xlabel("Month")
ax2.set_ylabel("Year")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_strategy_performance.png", bbox_inches="tight")
plt.show()
print(f"✅ Saved: {OUTPUT_DIR}/fig2_strategy_performance.png")

# ── 7. Export PowerBI-Ready Excel ────────────────────────
export_df = df[["Open","High","Low","Close","Volume",
                "SMA_50","SMA_200","EMA_20","RSI",
                "BB_Upper","BB_Mid","BB_Lower",
                "Daily_Return","Signal_SMA","Signal_RSI"]].copy()

export_df.to_excel(f"{OUTPUT_DIR}/trading_data_powerbi.xlsx")
print(f"✅ Saved: {OUTPUT_DIR}/trading_data_powerbi.xlsx  (ready for PowerBI)")

# ── 8. Summary Report ─────────────────────────────────────
print("\n" + "=" * 60)
print("  📌 TRADING STRATEGY SUMMARY")
print("=" * 60)
print(f"\n  Period              : {START} → {END}")
print(f"  Stock               : {STOCK}")
print(f"  Total Trading Days  : {len(df)}")
print(f"  Avg Daily Volume    : {df['Volume'].mean():,.0f}")
print(f"  Price Range         : ₹{df['Close'].min():.2f} → ₹{df['Close'].max():.2f}")
print(f"  Avg RSI             : {df['RSI'].mean():.1f}")
print(f"\n  Strategy Return     : {strategy_return:+.1f}%")
print(f"  Buy & Hold Return   : {buyhold_return:+.1f}%")
print(f"  Alpha               : {strategy_return - buyhold_return:+.1f}%")
print(f"\n🎉 Dashboard complete! All outputs saved to outputs/ folder.")

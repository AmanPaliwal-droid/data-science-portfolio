"""
=============================================================
E-Commerce Sales EDA — Uncovering Hidden Patterns
=============================================================
Author      : Aman Paliwal
Skills Used : Python, Pandas, NumPy, Matplotlib, Seaborn,
              Statistical Analysis, Feature Engineering
=============================================================

Story: Suppose I'm a data analyst at Flipkart and management wants to know:
  1. Which product categories drive the most revenue?
  2. When do customers buy most? (time-of-day, day-of-week)
  3. Which cities are underperforming?
  4. Are discounts actually helping sales?
  5. Who are our high-value customer segments?
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = ["#03045E","#0077B6","#00B4D8","#90E0EF","#CAF0F8",
           "#FF6B35","#F7C59F","#EFEFD0","#004E89","#1A936F"]
plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})
sns.set_theme(style="whitegrid")

# ── 1. Generate Realistic E-Commerce Dataset ──────────────
np.random.seed(21)
N = 5000

print("=" * 60)
print("  FLIPMART E-COMMERCE SALES EDA")
print("=" * 60)

CATEGORIES = {
    "Electronics"   : (8000, 50000, 0.12),
    "Fashion"       : (500,  4000,  0.35),
    "Home & Kitchen": (1000, 15000, 0.20),
    "Books"         : (200,  2000,  0.05),
    "Sports"        : (800,  8000,  0.15),
    "Beauty"        : (300,  3000,  0.28),
    "Toys"          : (400,  4000,  0.22),
    "Grocery"       : (100,  1500,  0.08),
}
CITIES = {
    "Mumbai"    : 0.18, "Delhi"     : 0.17, "Bengaluru" : 0.15,
    "Hyderabad" : 0.12, "Chennai"   : 0.10, "Pune"      : 0.09,
    "Kolkata"   : 0.08, "Indore"    : 0.05, "Jaipur"    : 0.04,
    "Ahmedabad" : 0.02,
}
PAYMENT  = ["UPI", "Credit Card", "Debit Card", "COD", "Net Banking"]
SEGMENTS = ["Bronze", "Silver", "Gold", "Platinum"]

cat_choices = list(CATEGORIES.keys())
cat_probs   = [0.20, 0.18, 0.15, 0.08, 0.12, 0.12, 0.08, 0.07]

categories  = np.random.choice(cat_choices, N, p=cat_probs)
cities      = np.random.choice(list(CITIES.keys()), N,
                               p=list(CITIES.values()))
payment     = np.random.choice(PAYMENT, N, p=[0.35,0.25,0.20,0.15,0.05])
segment     = np.random.choice(SEGMENTS, N, p=[0.40,0.30,0.20,0.10])

order_dates = pd.date_range("2023-01-01", "2024-12-31", periods=N)
order_dates = order_dates + pd.to_timedelta(
    np.random.randint(0, 86400, N), unit="s")

prices      = []
discounts   = []
for cat in categories:
    lo, hi, disc_rate = CATEGORIES[cat]
    price = np.random.uniform(lo, hi)
    disc  = round(np.random.uniform(0, disc_rate * 100), 1)
    prices.append(round(price, 2))
    discounts.append(disc)

prices    = np.array(prices)
discounts = np.array(discounts)
qty       = np.random.randint(1, 5, N)
revenue   = prices * qty * (1 - discounts / 100)

ratings   = np.random.choice([1,2,3,4,5], N, p=[0.05,0.08,0.15,0.42,0.30])
returned  = np.random.choice([0, 1], N, p=[0.88, 0.12])

df = pd.DataFrame({
    "order_id"    : [f"ORD{str(i).zfill(6)}" for i in range(1, N+1)],
    "order_date"  : order_dates,
    "category"    : categories,
    "city"        : cities,
    "payment_mode": payment,
    "segment"     : segment,
    "unit_price"  : prices,
    "quantity"    : qty,
    "discount_pct": discounts,
    "revenue"     : revenue.round(2),
    "rating"      : ratings,
    "returned"    : returned,
})

# Feature Engineering
df["order_hour"]    = df["order_date"].dt.hour
df["order_dow"]     = df["order_date"].dt.day_name()
df["order_month"]   = df["order_date"].dt.month_name()
df["order_quarter"] = df["order_date"].dt.quarter
df["order_year"]    = df["order_date"].dt.year
df["profit_margin"] = ((df["revenue"] - df["unit_price"] * df["quantity"] * 0.6)
                        / df["revenue"] * 100).round(1)

print(f"\n📋 Dataset Shape  : {df.shape}")
print(f"📅 Date Range     : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
print(f"💰 Total Revenue  : ₹{df['revenue'].sum():,.0f}")
print(f"📦 Total Orders   : {N:,}")
print(f"↩️  Return Rate    : {df['returned'].mean()*100:.1f}%")
print(f"⭐ Avg Rating     : {df['rating'].mean():.2f}")

# ── 2. EDA — Figure 1: Revenue Overview ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Figure 1 — Revenue Overview Dashboard", fontsize=15, fontweight="bold")

# Revenue by Category
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=True)
axes[0,0].barh(cat_rev.index, cat_rev.values / 1e6,
               color=PALETTE[:len(cat_rev)], edgecolor="white")
axes[0,0].set_title("Revenue by Category (₹ Millions)")
axes[0,0].set_xlabel("Revenue (₹M)")
for i, (val) in enumerate(cat_rev.values):
    axes[0,0].text(val/1e6 + 0.2, i, f"₹{val/1e6:.1f}M", va="center", fontsize=9)

# Revenue by Quarter & Year
q_rev = df.groupby(["order_year","order_quarter"])["revenue"].sum().unstack()
q_rev.T.plot(kind="bar", ax=axes[0,1], color=[PALETTE[0], PALETTE[5]],
             edgecolor="white", width=0.7)
axes[0,1].set_title("Quarterly Revenue by Year")
axes[0,1].set_xlabel("Quarter")
axes[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{x/1e6:.1f}M"))
axes[0,1].tick_params(axis="x", rotation=0)
axes[0,1].legend(title="Year")

# Payment Mode Pie
pay_rev = df.groupby("payment_mode")["revenue"].sum()
axes[1,0].pie(pay_rev, labels=pay_rev.index, autopct="%1.1f%%",
              colors=PALETTE[:5], startangle=90, pctdistance=0.75)
axes[1,0].set_title("Revenue Share by Payment Mode")

# Segment-wise Revenue
seg_order = ["Bronze","Silver","Gold","Platinum"]
seg_rev   = df.groupby("segment")["revenue"].agg(["sum","mean"])
bars = axes[1,1].bar(seg_order,
                     [seg_rev.loc[s,"sum"]/1e6 for s in seg_order],
                     color=[PALETTE[3], PALETTE[2], PALETTE[5], PALETTE[0]])
axes[1,1].set_title("Total Revenue by Customer Segment")
axes[1,1].set_ylabel("Revenue (₹M)")
for bar in bars:
    axes[1,1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.1,
                   f"₹{bar.get_height():.1f}M", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_revenue_overview.png", bbox_inches="tight")
plt.show()
print(f"\n✅ Saved: {OUTPUT_DIR}/fig1_revenue_overview.png")

# ── 3. EDA — Figure 2: Time Patterns ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Figure 2 — Time-Based Buying Patterns", fontsize=15, fontweight="bold")

# Hourly revenue heatmap
dow_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
hour_dow   = df.groupby(["order_dow","order_hour"])["revenue"].sum().unstack(fill_value=0)
hour_dow   = hour_dow.reindex(dow_order)
sns.heatmap(hour_dow / 1e3, ax=axes[0], cmap="YlOrRd",
            annot=False, linewidths=0, cbar_kws={"label":"Revenue (₹K)"})
axes[0].set_title("Revenue Heatmap: Day of Week × Hour")
axes[0].set_xlabel("Hour of Day")

# Monthly revenue trend
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
m_rev = (df[df["order_year"] == 2024]
         .groupby("order_month")["revenue"].sum()
         .reindex(month_order))
axes[1].plot(range(len(m_rev)), m_rev.values / 1e6,
             color=PALETTE[0], marker="o", lw=2.5)
axes[1].fill_between(range(len(m_rev)), m_rev.values / 1e6,
                     alpha=0.1, color=PALETTE[0])
axes[1].set_xticks(range(len(m_rev)))
axes[1].set_xticklabels([m[:3] for m in month_order], rotation=45)
axes[1].set_title("Monthly Revenue Trend (2024)")
axes[1].set_ylabel("Revenue (₹M)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_time_patterns.png", bbox_inches="tight")
plt.show()
print(f"✅ Saved: {OUTPUT_DIR}/fig2_time_patterns.png")

# ── 4. EDA — Figure 3: Discounts & Returns ───────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 3 — Discount Impact & Returns Analysis", fontsize=15, fontweight="bold")

# Discount vs Revenue scatter
df["disc_band"] = pd.cut(df["discount_pct"], bins=[0,5,15,25,36],
                          labels=["0-5%","5-15%","15-25%","25%+"])
disc_agg = df.groupby("disc_band")["revenue"].agg(["mean","sum"])
axes[0].bar(disc_agg.index, disc_agg["mean"],
            color=[PALETTE[0],PALETTE[2],PALETTE[5],PALETTE[6]])
axes[0].set_title("Avg Order Value by Discount Band")
axes[0].set_ylabel("Avg Revenue (₹)")
for i, val in enumerate(disc_agg["mean"]):
    axes[0].text(i, val + 50, f"₹{val:,.0f}", ha="center", fontsize=9)

# Return rate by category
ret_cat = df.groupby("category")["returned"].mean() * 100
ret_cat = ret_cat.sort_values(ascending=True)
axes[1].barh(ret_cat.index, ret_cat.values,
             color=[PALETTE[5] if v > 12 else PALETTE[1] for v in ret_cat.values])
axes[1].axvline(df["returned"].mean()*100, color="red",
                linestyle="--", lw=1.5, label=f'Avg: {df["returned"].mean()*100:.1f}%')
axes[1].set_title("Return Rate by Category (%)")
axes[1].set_xlabel("Return Rate (%)")
axes[1].legend()

# Rating distribution by segment
seg_rat = df.groupby(["segment","rating"]).size().unstack(fill_value=0)
seg_rat = seg_rat.reindex(seg_order)
seg_rat.plot(kind="bar", stacked=True, ax=axes[2],
             color=PALETTE[1:6], edgecolor="white", width=0.6)
axes[2].set_title("Rating Distribution by Segment")
axes[2].set_xlabel("Segment")
axes[2].tick_params(axis="x", rotation=0)
axes[2].legend(title="Rating", loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_discounts_returns.png", bbox_inches="tight")
plt.show()
print(f"✅ Saved: {OUTPUT_DIR}/fig3_discounts_returns.png")

# ── 5. EDA — Figure 4: City Performance ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 4 — City-wise Performance Analysis", fontsize=15, fontweight="bold")

city_stats = df.groupby("city").agg(
    total_revenue=("revenue","sum"),
    order_count  =("order_id","count"),
    avg_order    =("revenue","mean"),
    return_rate  =("returned","mean")
).reset_index()
city_stats["total_revenue_M"] = city_stats["total_revenue"] / 1e6
city_stats = city_stats.sort_values("total_revenue_M", ascending=False)

sc = axes[0].scatter(city_stats["order_count"],
                     city_stats["total_revenue_M"],
                     s=city_stats["avg_order"] / 20,
                     c=city_stats["return_rate"] * 100,
                     cmap="RdYlGn_r", alpha=0.8, edgecolors="black", linewidth=0.5)
plt.colorbar(sc, ax=axes[0], label="Return Rate (%)")
for _, row in city_stats.iterrows():
    axes[0].annotate(row["city"],
                     (row["order_count"], row["total_revenue_M"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[0].set_xlabel("Order Count")
axes[0].set_ylabel("Total Revenue (₹M)")
axes[0].set_title("City Bubble Chart (size = Avg Order Value)")

# Horizontal bar of avg order value
bars = axes[1].barh(city_stats["city"], city_stats["avg_order"],
                    color=[PALETTE[0] if v > city_stats["avg_order"].mean()
                           else PALETTE[5] for v in city_stats["avg_order"]])
axes[1].axvline(city_stats["avg_order"].mean(), color="gray",
                linestyle="--", label="Overall Average")
axes[1].set_title("Average Order Value by City")
axes[1].set_xlabel("Avg Order Value (₹)")
axes[1].legend()
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_city_performance.png", bbox_inches="tight")
plt.show()
print(f"✅ Saved: {OUTPUT_DIR}/fig4_city_performance.png")

# ── 6. Statistical Tests ──────────────────────────────────
print("\n🔬 Step 5: Statistical Significance Tests")

# Do Gold/Platinum customers spend more than Bronze?
gold_rev   = df[df["segment"].isin(["Gold","Platinum"])]["revenue"]
bronze_rev = df[df["segment"] == "Bronze"]["revenue"]
t_stat, p_val = stats.ttest_ind(gold_rev, bronze_rev)
print(f"\n   T-test (Gold+Platinum vs Bronze revenue):")
print(f"   t-stat = {t_stat:.2f}, p-value = {p_val:.4f}")
print(f"   → {'Significant difference ✅' if p_val < 0.05 else 'No significant difference ❌'}")

# Correlation between discount and rating
r, p = stats.pearsonr(df["discount_pct"], df["rating"])
print(f"\n   Pearson Correlation (Discount % vs Rating):")
print(f"   r = {r:.4f}, p = {p:.4f}")
print(f"   → {'Significant ✅' if p < 0.05 else 'Not significant ❌'}")

# ── 7. Export ─────────────────────────────────────────────
with pd.ExcelWriter(f"{OUTPUT_DIR}/ecommerce_eda_export.xlsx", engine="openpyxl") as xl:
    df.drop(columns=["disc_band"]).to_excel(xl, sheet_name="Raw_Data",     index=False)
    df.groupby("category").agg(
        total_revenue=("revenue","sum"), orders=("order_id","count"),
        avg_discount=("discount_pct","mean"), return_rate=("returned","mean")
    ).reset_index().to_excel(xl, sheet_name="Category_Summary", index=False)
    city_stats.to_excel(xl, sheet_name="City_Summary", index=False)

print(f"\n✅ Saved: {OUTPUT_DIR}/ecommerce_eda_export.xlsx")

# ── 8. Key Insights ───────────────────────────────────────
print("\n" + "=" * 60)
print("  📌 KEY BUSINESS INSIGHTS")
print("=" * 60)
top_cat = df.groupby("category")["revenue"].sum().idxmax()
top_city = city_stats.iloc[0]["city"]
peak_hour = df.groupby("order_hour")["revenue"].sum().idxmax()
print(f"\n1. 🏆 Top Revenue Category  : {top_cat}")
print(f"2. 🌆 Top Performing City   : {top_city}")
print(f"3. ⏰ Peak Shopping Hour    : {peak_hour}:00 hrs")
print(f"4. 💳 Most Used Payment     : {df.groupby('payment_mode')['revenue'].sum().idxmax()}")
print(f"5. 📊 High Discount ≠ High Rating (r={r:.2f})")
print(f"6. 🏅 Platinum customers spend {df[df['segment']=='Platinum']['revenue'].mean()/df[df['segment']=='Bronze']['revenue'].mean():.1f}x more than Bronze")
print("\n🎉 EDA Complete! All figures and exports saved to outputs/")

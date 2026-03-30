"""
=============================================================
PROJECT 1: Student Performance Analysis (Ram's Story)
=============================================================
Author      : [Your Name]
Role Target : Instructional Associate @ CodingGita
Skills Used : Python, Pandas, Matplotlib, Seaborn, NumPy, EDA
JD Match    : EDA sessions, data-driven insights, student outcomes
=============================================================

Story: Ram is a data analyst at an edtech company. His manager
asks him to uncover WHY students are failing and WHAT interventions
can improve outcomes. This script walks through that journey.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = ["#2D6A4F", "#40916C", "#74C69D", "#D62828", "#F4A261"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})

# ── 1. Data Generation (simulating real student dataset) ──
np.random.seed(42)

N = 500

def generate_student_data(n):
    study_hours   = np.random.normal(5, 2, n).clip(0, 12)
    attendance    = np.random.normal(75, 15, n).clip(30, 100)
    prev_score    = np.random.normal(60, 15, n).clip(20, 100)
    sleep_hours   = np.random.normal(7, 1.5, n).clip(4, 10)
    part_time_job = np.random.choice([0, 1], n, p=[0.65, 0.35])
    gender        = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    section       = np.random.choice(["A", "B", "C", "D"], n)
    internet      = np.random.choice([0, 1], n, p=[0.2, 0.8])

    # Score formula with realistic noise
    score = (
        0.30 * study_hours * 8
        + 0.25 * attendance * 0.7
        + 0.20 * prev_score * 0.6
        + 0.10 * sleep_hours * 5
        - 0.10 * part_time_job * 12
        + 0.05 * internet * 8
        + np.random.normal(0, 6, n)
    ).clip(0, 100)

    grade = pd.cut(score,
                   bins=[0, 40, 55, 70, 85, 100],
                   labels=["F", "D", "C", "B", "A"])

    return pd.DataFrame({
        "student_id"   : [f"STU{str(i).zfill(4)}" for i in range(1, n+1)],
        "name"         : [f"Student_{i}" for i in range(1, n+1)],
        "gender"       : gender,
        "section"      : section,
        "study_hours"  : study_hours.round(1),
        "attendance_pct": attendance.round(1),
        "prev_score"   : prev_score.round(1),
        "sleep_hours"  : sleep_hours.round(1),
        "part_time_job": part_time_job,
        "internet_access": internet,
        "final_score"  : score.round(1),
        "grade"        : grade
    })

# ── 2. Load & Inspect ─────────────────────────────────────
print("=" * 60)
print("  RAM'S STUDENT PERFORMANCE ANALYSIS")
print("=" * 60)

df = generate_student_data(N)

print(f"\n📋 Dataset Shape  : {df.shape}")
print(f"📌 Columns        : {list(df.columns)}")
print(f"\n📊 First 5 Rows:")
print(df.head())

print(f"\n🔍 Basic Statistics:")
print(df.describe().round(2))

print(f"\n❓ Missing Values:")
print(df.isnull().sum())

print(f"\n🎓 Grade Distribution:")
print(df["grade"].value_counts().sort_index())

# ── 3. EDA — Figure 1: Score Distributions ───────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure 1 — Score & Grade Distributions", fontsize=14, fontweight="bold")

# Histogram of final scores
axes[0].hist(df["final_score"], bins=25, color="#2D6A4F", edgecolor="white")
axes[0].axvline(df["final_score"].mean(), color="#D62828", linestyle="--", label=f'Mean: {df["final_score"].mean():.1f}')
axes[0].set_title("Final Score Distribution")
axes[0].set_xlabel("Score")
axes[0].legend()

# Grade pie chart
grade_counts = df["grade"].value_counts().sort_index()
axes[1].pie(grade_counts, labels=grade_counts.index, autopct="%1.1f%%",
            colors=PALETTE, startangle=90)
axes[1].set_title("Grade Breakdown")

# Score by gender boxplot
df.boxplot(column="final_score", by="gender", ax=axes[2],
           boxprops=dict(color="#2D6A4F"),
           medianprops=dict(color="#D62828", linewidth=2))
axes[2].set_title("Score by Gender")
axes[2].set_xlabel("Gender")
plt.suptitle("")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_distributions.png", bbox_inches="tight")
plt.show()
print(f"\n✅ Saved: {OUTPUT_DIR}/fig1_distributions.png")

# ── 4. EDA — Figure 2: Study Habits vs Performance ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 2 — Study Habits vs Performance", fontsize=14, fontweight="bold")

# Scatter: study hours vs final score
sc = axes[0].scatter(df["study_hours"], df["final_score"],
                     c=df["attendance_pct"], cmap="YlGn",
                     alpha=0.6, edgecolors="none", s=40)
plt.colorbar(sc, ax=axes[0], label="Attendance %")
m, b, r, p, _ = stats.linregress(df["study_hours"], df["final_score"])
x_line = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 100)
axes[0].plot(x_line, m * x_line + b, color="#D62828", linewidth=2,
             label=f"r = {r:.2f}, p = {p:.3f}")
axes[0].set_title("Study Hours vs Final Score")
axes[0].set_xlabel("Study Hours/Day")
axes[0].set_ylabel("Final Score")
axes[0].legend()

# Attendance buckets
df["attendance_band"] = pd.cut(df["attendance_pct"],
                                bins=[0, 60, 75, 90, 100],
                                labels=["<60%", "60-75%", "75-90%", "90%+"])
avg_by_att = df.groupby("attendance_band")["final_score"].mean()
bars = axes[1].bar(avg_by_att.index, avg_by_att.values,
                   color=["#D62828", "#F4A261", "#74C69D", "#2D6A4F"])
for bar, val in zip(bars, avg_by_att.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", fontsize=10)
axes[1].set_title("Avg Score by Attendance Band")
axes[1].set_xlabel("Attendance Band")
axes[1].set_ylabel("Avg Final Score")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_study_habits.png", bbox_inches="tight")
plt.show()
print(f"✅ Saved: {OUTPUT_DIR}/fig2_study_habits.png")

# ── 5. EDA — Figure 3: Risk Factors ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 3 — Risk Factors Impacting Performance", fontsize=14, fontweight="bold")

# Part-time job impact
job_labels = {0: "No Job", 1: "Part-Time Job"}
df["job_label"] = df["part_time_job"].map(job_labels)
sns.boxplot(data=df, x="job_label", y="final_score",
            palette={"No Job": "#40916C", "Part-Time Job": "#D62828"},
            ax=axes[0])
axes[0].set_title("Part-Time Job vs Score")
axes[0].set_xlabel("")

# Correlation heatmap
numeric_cols = ["study_hours", "attendance_pct", "prev_score",
                "sleep_hours", "part_time_job", "internet_access", "final_score"]
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, ax=axes[1],
            linewidths=0.5, square=True)
axes[1].set_title("Correlation Matrix")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_risk_factors.png", bbox_inches="tight")
plt.show()
print(f"✅ Saved: {OUTPUT_DIR}/fig3_risk_factors.png")

# ── 6. EDA — Figure 4: Section-wise Performance ──────────
fig, ax = plt.subplots(figsize=(10, 5))
section_grade = df.groupby(["section", "grade"]).size().unstack(fill_value=0)
section_grade.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.75)
ax.set_title("Section-wise Grade Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Section")
ax.set_ylabel("Number of Students")
ax.legend(title="Grade", loc="upper right")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_section_grades.png", bbox_inches="tight")
plt.show()
print(f"✅ Saved: {OUTPUT_DIR}/fig4_section_grades.png")

# ── 7. Key Insights & Recommendations ────────────────────
print("\n" + "=" * 60)
print("  📌 RAM'S KEY FINDINGS & RECOMMENDATIONS")
print("=" * 60)

fail_rate = (df["grade"] == "F").mean() * 100
high_att_high_score = df[df["attendance_pct"] >= 90]["final_score"].mean()
low_att_score       = df[df["attendance_pct"] < 60]["final_score"].mean()
job_penalty         = df[df["part_time_job"] == 1]["final_score"].mean() - \
                      df[df["part_time_job"] == 0]["final_score"].mean()

print(f"\n1. 📉 Failure Rate        : {fail_rate:.1f}% of students scored below 40")
print(f"2. 🎯 Attendance Matters  : 90%+ attendance → avg score {high_att_high_score:.1f}")
print(f"                           <60% attendance → avg score {low_att_score:.1f}")
print(f"3. 💼 Job Penalty         : Part-time jobs reduce avg score by {abs(job_penalty):.1f} pts")
print(f"4. 📚 Study Correlation   : Pearson r = {df['study_hours'].corr(df['final_score']):.2f} (study hrs vs score)")
print(f"5. 😴 Sleep Impact        : Pearson r = {df['sleep_hours'].corr(df['final_score']):.2f} (sleep vs score)")

print("\n💡 Recommendations for Instructors:")
print("   → Flag students with attendance < 65% early in the semester")
print("   → Offer recorded sessions for students with part-time jobs")
print("   → Create 'Study Buddy' groups to boost study hours")
print("   → Use prev_score to personalize support during onboarding")

# ── 8. Export clean dataset ───────────────────────────────
df.drop(columns=["job_label", "attendance_band"]).to_csv(
    f"{OUTPUT_DIR}/student_data_clean.csv", index=False)
print(f"\n✅ Clean dataset saved: {OUTPUT_DIR}/student_data_clean.csv")
print("\n🎉 Analysis complete! Check the 'outputs/' folder for all charts.")

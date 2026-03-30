"""
=============================================================
PROJECT 3: SQL + Python Data Pipeline
=============================================================
Author      : [Your Name]
Role Target : Instructional Associate @ CodingGita
Skills Used : Python, SQLite, Pandas, SQL (DDL, DML, aggregations,
              joins, window functions, CTEs), Matplotlib
JD Match    : SQL for data retrieval, manipulation, data science projects
=============================================================

Story: You are the data engineer for an e-learning platform.
Students enroll in courses, complete lessons, and earn certificates.
Your job is to build the database, populate it, and run analytical
SQL queries to answer real business questions from the management team.
=============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────
DB_PATH    = "elearning_platform.db"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
plt.rcParams.update({"figure.dpi": 120})

print("=" * 60)
print("  SQL + PYTHON DATA PIPELINE")
print("  E-Learning Platform Analytics")
print("=" * 60)

# ── 1. Schema Creation (DDL) ──────────────────────────────
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

print("\n📐 Step 1: Creating database schema...")

cur.executescript("""
    DROP TABLE IF EXISTS enrollments;
    DROP TABLE IF EXISTS lesson_progress;
    DROP TABLE IF EXISTS certificates;
    DROP TABLE IF EXISTS courses;
    DROP TABLE IF EXISTS students;

    CREATE TABLE students (
        student_id   INTEGER PRIMARY KEY,
        name         TEXT NOT NULL,
        email        TEXT UNIQUE NOT NULL,
        city         TEXT,
        age          INTEGER,
        signup_date  DATE,
        plan         TEXT CHECK(plan IN ('Free','Pro','Enterprise'))
    );

    CREATE TABLE courses (
        course_id    INTEGER PRIMARY KEY,
        title        TEXT NOT NULL,
        category     TEXT,
        difficulty   TEXT CHECK(difficulty IN ('Beginner','Intermediate','Advanced')),
        duration_hrs REAL,
        price_inr    REAL
    );

    CREATE TABLE enrollments (
        enrollment_id INTEGER PRIMARY KEY,
        student_id    INTEGER REFERENCES students(student_id),
        course_id     INTEGER REFERENCES courses(course_id),
        enrolled_date DATE,
        completed     INTEGER DEFAULT 0,
        completion_date DATE,
        rating        REAL
    );

    CREATE TABLE lesson_progress (
        progress_id  INTEGER PRIMARY KEY,
        enrollment_id INTEGER REFERENCES enrollments(enrollment_id),
        lesson_no    INTEGER,
        watched_pct  REAL,
        watch_date   DATE
    );

    CREATE TABLE certificates (
        cert_id      INTEGER PRIMARY KEY,
        student_id   INTEGER REFERENCES students(student_id),
        course_id    INTEGER REFERENCES courses(course_id),
        issued_date  DATE,
        score        REAL
    );
""")
conn.commit()
print("   ✅ Tables created: students, courses, enrollments, lesson_progress, certificates")

# ── 2. Data Population (DML INSERT) ──────────────────────
print("\n📥 Step 2: Populating tables with sample data...")

cities    = ["Mumbai","Delhi","Bengaluru","Hyderabad","Pune","Chennai","Kolkata","Indore"]
plans     = ["Free","Pro","Enterprise"]
plan_wts  = [0.5, 0.35, 0.15]

students_data = []
for i in range(1, 201):
    students_data.append((
        i,
        f"Student_{i}",
        f"student{i}@email.com",
        np.random.choice(cities),
        int(np.random.randint(18, 45)),
        str(pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(np.random.randint(0, 730)))),
        np.random.choice(plans, p=plan_wts)
    ))
cur.executemany("INSERT INTO students VALUES (?,?,?,?,?,?,?)", students_data)

courses_data = [
    (1,  "Python for Data Science",    "Data Science",  "Beginner",     30, 2999),
    (2,  "Advanced SQL Mastery",        "Database",      "Intermediate", 20, 1999),
    (3,  "Machine Learning A-Z",        "AI/ML",         "Advanced",     50, 4999),
    (4,  "PowerBI Dashboarding",        "Analytics",     "Beginner",     15, 1499),
    (5,  "Data Structures & Algo",      "Programming",   "Intermediate", 25, 2499),
    (6,  "Deep Learning with TF",       "AI/ML",         "Advanced",     45, 5999),
    (7,  "Excel for Business Analysis", "Analytics",     "Beginner",     10,  999),
    (8,  "Web Scraping with Python",    "Programming",   "Intermediate", 12, 1299),
    (9,  "Statistics for Data Science", "Data Science",  "Intermediate", 18, 1799),
    (10, "NLP Fundamentals",            "AI/ML",         "Advanced",     35, 3999),
]
cur.executemany("INSERT INTO courses VALUES (?,?,?,?,?,?)", courses_data)

# Enrollments
enrollment_id = 1
enroll_rows   = []
cert_rows     = []
cert_id       = 1

for sid in range(1, 201):
    num_courses = int(np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
    chosen      = np.random.choice(range(1, 11), size=num_courses, replace=False)
    for cid in chosen:
        enr_date    = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(np.random.randint(0, 700)))
        completed   = int(np.random.choice([0, 1], p=[0.4, 0.6]))
        comp_date   = str(enr_date + pd.Timedelta(days=int(np.random.randint(10, 90)))) if completed else None
        rating      = round(np.random.uniform(3.0, 5.0), 1) if completed else None
        enroll_rows.append((enrollment_id, sid, cid, str(enr_date.date()),
                            completed, comp_date, rating))
        if completed:
            score = round(np.random.uniform(60, 100), 1)
            cert_rows.append((cert_id, sid, cid, comp_date, score))
            cert_id += 1
        enrollment_id += 1

cur.executemany("INSERT INTO enrollments VALUES (?,?,?,?,?,?,?)", enroll_rows)
cur.executemany("INSERT INTO certificates VALUES (?,?,?,?,?)", cert_rows)
conn.commit()

print(f"   ✅ Students     : {len(students_data)}")
print(f"   ✅ Courses      : {len(courses_data)}")
print(f"   ✅ Enrollments  : {len(enroll_rows)}")
print(f"   ✅ Certificates : {len(cert_rows)}")

# ── 3. SQL Analytical Queries ─────────────────────────────
print("\n🔍 Step 3: Running SQL analytical queries...\n")

# ── Q1: Enrollment & Completion by Course ────────────────
q1 = """
SELECT
    c.title                              AS course_title,
    c.category,
    c.difficulty,
    COUNT(e.enrollment_id)               AS total_enrolled,
    SUM(e.completed)                     AS total_completed,
    ROUND(
        100.0 * SUM(e.completed) / COUNT(e.enrollment_id), 1
    )                                    AS completion_rate_pct,
    ROUND(AVG(e.rating), 2)              AS avg_rating
FROM enrollments e
JOIN courses c ON e.course_id = c.course_id
GROUP BY c.course_id
ORDER BY completion_rate_pct DESC;
"""
df_q1 = pd.read_sql_query(q1, conn)
print("📊 Q1 — Course Completion Rates:")
print(df_q1.to_string(index=False))

# ── Q2: Top Students by Certificates (Window Function) ───
q2 = """
SELECT
    s.name,
    s.city,
    s.plan,
    COUNT(cert.cert_id)                     AS total_certs,
    ROUND(AVG(cert.score), 1)               AS avg_cert_score,
    RANK() OVER (ORDER BY COUNT(cert.cert_id) DESC) AS rank
FROM students s
LEFT JOIN certificates cert ON s.student_id = cert.student_id
GROUP BY s.student_id
HAVING total_certs > 0
ORDER BY rank
LIMIT 10;
"""
df_q2 = pd.read_sql_query(q2, conn)
print("\n🏆 Q2 — Top 10 Students by Certificates Earned:")
print(df_q2.to_string(index=False))

# ── Q3: Revenue by Category (CTE) ─────────────────────────
q3 = """
WITH revenue_cte AS (
    SELECT
        c.category,
        c.price_inr,
        e.completed,
        c.course_id
    FROM enrollments e
    JOIN courses c ON e.course_id = c.course_id
    WHERE e.completed = 1
)
SELECT
    category,
    COUNT(*)                                AS completions,
    ROUND(SUM(price_inr), 0)               AS total_revenue_inr,
    ROUND(AVG(price_inr), 0)               AS avg_revenue_per_completion
FROM revenue_cte
GROUP BY category
ORDER BY total_revenue_inr DESC;
"""
df_q3 = pd.read_sql_query(q3, conn)
print("\n💰 Q3 — Revenue by Category (CTE):")
print(df_q3.to_string(index=False))

# ── Q4: Monthly Enrollment Trend ──────────────────────────
q4 = """
SELECT
    SUBSTR(enrolled_date, 1, 7)   AS month,
    COUNT(*)                       AS new_enrollments,
    SUM(completed)                 AS completions
FROM enrollments
GROUP BY month
ORDER BY month;
"""
df_q4 = pd.read_sql_query(q4, conn)
print("\n📅 Q4 — Monthly Enrollment Trend (first 6 months):")
print(df_q4.head(6).to_string(index=False))

# ── Q5: Plan-wise Performance (subquery) ──────────────────
q5 = """
SELECT
    s.plan,
    COUNT(DISTINCT s.student_id)    AS total_students,
    ROUND(AVG(sub.cert_count), 1)   AS avg_certs_per_student,
    ROUND(AVG(sub.avg_score), 1)    AS avg_score
FROM students s
LEFT JOIN (
    SELECT student_id,
           COUNT(*)       AS cert_count,
           AVG(score)     AS avg_score
    FROM certificates
    GROUP BY student_id
) sub ON s.student_id = sub.student_id
GROUP BY s.plan
ORDER BY avg_certs_per_student DESC;
"""
df_q5 = pd.read_sql_query(q5, conn)
print("\n💼 Q5 — Performance by Subscription Plan:")
print(df_q5.to_string(index=False))

# ── 4. Visualisations ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("E-Learning Platform SQL Analytics Dashboard", fontsize=15, fontweight="bold")

# Chart 1 — Completion Rate by Course
ax1 = axes[0, 0]
bars = ax1.barh(df_q1["course_title"].str[:25], df_q1["completion_rate_pct"],
                color=PALETTE[1], edgecolor="white")
for bar, val in zip(bars, df_q1["completion_rate_pct"]):
    ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val}%", va="center", fontsize=9)
ax1.set_title("Completion Rate by Course (%)")
ax1.set_xlabel("Completion Rate (%)")
ax1.invert_yaxis()

# Chart 2 — Revenue by Category
ax2 = axes[0, 1]
wedges, texts, autotexts = ax2.pie(
    df_q3["total_revenue_inr"],
    labels=df_q3["category"],
    autopct="%1.1f%%",
    colors=PALETTE,
    startangle=90,
    pctdistance=0.75
)
ax2.set_title("Revenue Share by Category")

# Chart 3 — Monthly Enrollment Trend
ax3 = axes[1, 0]
ax3.plot(df_q4["month"], df_q4["new_enrollments"], color=PALETTE[0],
         marker="o", lw=2, label="Enrollments")
ax3.plot(df_q4["month"], df_q4["completions"], color=PALETTE[2],
         marker="s", lw=2, linestyle="--", label="Completions")
ax3.set_title("Monthly Enrollment & Completion Trend")
ax3.set_xlabel("Month")
ax3.tick_params(axis="x", rotation=45)
ax3.legend()
ax3.grid(alpha=0.3)

# Chart 4 — Plan-wise Avg Certificates
ax4 = axes[1, 1]
bars4 = ax4.bar(df_q5["plan"], df_q5["avg_certs_per_student"],
                color=[PALETTE[2], PALETTE[3], PALETTE[4]])
for bar, val in zip(bars4, df_q5["avg_certs_per_student"]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
ax4.set_title("Avg Certificates per Student by Plan")
ax4.set_ylabel("Avg Certificates")
ax4.set_ylim(0, df_q5["avg_certs_per_student"].max() + 0.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sql_analytics_dashboard.png", bbox_inches="tight")
plt.show()
print(f"\n✅ Saved: {OUTPUT_DIR}/sql_analytics_dashboard.png")

# ── 5. Export PowerBI Excel ───────────────────────────────
with pd.ExcelWriter(f"{OUTPUT_DIR}/sql_pipeline_export.xlsx", engine="openpyxl") as writer:
    df_q1.to_excel(writer, sheet_name="Course_Completion", index=False)
    df_q2.to_excel(writer, sheet_name="Top_Students",     index=False)
    df_q3.to_excel(writer, sheet_name="Revenue_Category", index=False)
    df_q4.to_excel(writer, sheet_name="Monthly_Trend",    index=False)
    df_q5.to_excel(writer, sheet_name="Plan_Performance", index=False)

print(f"✅ Saved: {OUTPUT_DIR}/sql_pipeline_export.xlsx  (5 sheets, PowerBI ready)")

conn.close()
os.remove(DB_PATH)  # clean up temp DB

print("\n" + "=" * 60)
print("  🎉 Pipeline Complete!")
print("  SQL skills demonstrated: DDL, DML, JOINs, GROUP BY,")
print("  WINDOW FUNCTIONS (RANK), CTEs, Subqueries, HAVING")
print("=" * 60)

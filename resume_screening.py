"""
=============================================================
  Resume Screening & Candidate Ranking System
  Task 3 — Future Interns ML Project
=============================================================
  Features:
    - Synthetic resume + job-description dataset (no download needed)
    - Text cleaning & preprocessing (NLTK)
    - Skill extraction using keyword NLP
    - TF-IDF cosine similarity (resume ↔ job description)
    - Weighted composite scoring
    - Candidate ranking with fit labels
    - Skill gap report per candidate
    - 6-panel evaluation dashboard (saved as PNG)
=============================================================
"""

# ──────────────────────────────────────────────
# 1.  IMPORTS
# ──────────────────────────────────────────────
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
warnings.filterwarnings("ignore")

# NLTK
import nltk
for pkg in ("stopwords", "punkt", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# ──────────────────────────────────────────────
# 2.  JOB DESCRIPTION
# ──────────────────────────────────────────────
JOB_ROLE = "Data Scientist"

JOB_DESCRIPTION = """
We are looking for a skilled Data Scientist to join our AI team.
The ideal candidate should have strong experience in Python programming,
machine learning, deep learning, and data analysis.

Required Skills:
Python, machine learning, data analysis, pandas, numpy, scikit-learn,
statistics, SQL, data visualization, matplotlib, feature engineering,
model evaluation, regression, classification, problem solving.

Preferred Skills:
TensorFlow, PyTorch, deep learning, neural networks, NLP, natural language
processing, computer vision, big data, Spark, cloud computing, AWS, Docker,
Git, A/B testing, hypothesis testing, communication, teamwork.

Experience: 2+ years in data science or related field.
Education: Bachelor's or Master's in Computer Science, Statistics, or related.
"""

# Required & preferred split for weighted scoring
REQUIRED_SKILLS = [
    "python", "machine learning", "data analysis", "pandas", "numpy",
    "scikit-learn", "statistics", "sql", "data visualization", "matplotlib",
    "feature engineering", "model evaluation", "regression", "classification",
]
PREFERRED_SKILLS = [
    "tensorflow", "pytorch", "deep learning", "neural networks", "nlp",
    "natural language processing", "computer vision", "spark", "aws",
    "docker", "git", "a/b testing", "hypothesis testing",
]

ALL_SKILLS = REQUIRED_SKILLS + PREFERRED_SKILLS


# ──────────────────────────────────────────────
# 3.  SYNTHETIC RESUME DATASET
# ──────────────────────────────────────────────
RESUMES = [
    {
        "name": "Anika Sharma",
        "text": """
        Experienced Data Scientist with 4 years at a fintech startup.
        Proficient in Python, pandas, numpy, scikit-learn, and matplotlib.
        Built regression and classification models for credit risk scoring.
        Strong background in statistics, SQL, and feature engineering.
        Familiar with TensorFlow and deep learning for fraud detection.
        Experience with Git, Docker, and AWS cloud deployment.
        Completed A/B testing for product feature rollouts.
        MSc in Statistics from IIT Delhi.
        """,
    },
    {
        "name": "Ravi Patel",
        "text": """
        Junior data analyst with 1 year of experience in Python and Excel.
        Basic knowledge of pandas and data visualization using matplotlib.
        Completed online courses in machine learning and statistics.
        Familiar with SQL for data querying.
        Built a classification project for college using scikit-learn.
        No industry experience in model evaluation or feature engineering.
        BSc in Computer Science, 2023 graduate.
        """,
    },
    {
        "name": "Meera Krishnan",
        "text": """
        ML Engineer with 3 years of experience in deep learning and NLP.
        Expert in PyTorch, TensorFlow, and neural networks.
        Natural language processing projects including text classification
        and sentiment analysis. Strong Python, pandas, numpy skills.
        SQL proficiency and data analysis experience. Git and Docker user.
        Hypothesis testing and A/B testing for model experiments.
        Published research in computer vision and image recognition.
        BTech from NIT Trichy, MSc from IISc Bangalore.
        """,
    },
    {
        "name": "Arjun Mehta",
        "text": """
        Senior Data Scientist with 6 years building scalable ML pipelines.
        Deep expertise in Python, scikit-learn, machine learning, statistics.
        Extensive feature engineering and model evaluation experience.
        Built regression and classification systems for e-commerce.
        Data visualization with matplotlib and Tableau. SQL and big data Spark.
        AWS certified, Docker, Git. Led A/B testing and hypothesis testing.
        Mentored junior data scientists. Strong communication and teamwork.
        PhD in Computer Science, specialization in data analysis.
        """,
    },
    {
        "name": "Priya Nair",
        "text": """
        Frontend developer transitioning into data science.
        Knows Python basics and has completed a machine learning bootcamp.
        Limited experience with pandas, numpy, and data analysis.
        No SQL, statistics, or model evaluation background.
        Interested in data visualization and learning matplotlib.
        No professional ML project experience yet.
        BSc in Information Technology.
        """,
    },
    {
        "name": "Karan Gupta",
        "text": """
        Data Scientist with 2 years in healthcare analytics.
        Python, pandas, numpy, matplotlib, scikit-learn daily tools.
        Built classification and regression models for patient outcomes.
        Statistics, SQL, data analysis, and feature engineering skills.
        Familiar with NLP for medical text. Git and basic AWS usage.
        Hypothesis testing for clinical research projects.
        MSc in Data Science, BioInformatics specialization.
        """,
    },
    {
        "name": "Sneha Joshi",
        "text": """
        Research analyst with strong background in statistics and data analysis.
        Python scripting, SQL, pandas for data wrangling and reporting.
        Familiar with machine learning concepts but limited hands-on scikit-learn.
        Data visualization with matplotlib and seaborn. Feature engineering basics.
        Model evaluation understanding from academic coursework.
        No deep learning, Docker, or cloud experience.
        MA in Economics with minor in Statistics.
        """,
    },
    {
        "name": "Dev Verma",
        "text": """
        DevOps Engineer moving into data science. Strong Git, Docker, AWS skills.
        Python programming and basic pandas knowledge.
        Completed Coursera machine learning certification by Andrew Ng.
        Understanding of classification concepts. Limited statistics background.
        SQL basics. No professional data science or model evaluation project.
        BE in Electronics and Communication Engineering.
        """,
    },
]


# ──────────────────────────────────────────────
# 4.  TEXT PREPROCESSING
# ──────────────────────────────────────────────
_stop = set(stopwords.words("english"))
_lem  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s/+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [_lem.lemmatize(t) for t in text.split()
              if t not in _stop and len(t) > 2]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# 5.  SKILL EXTRACTION
# ──────────────────────────────────────────────
def extract_skills(text: str, skill_list: list) -> list:
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        # match whole phrase
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


# ──────────────────────────────────────────────
# 6.  SCORING ENGINE
# ──────────────────────────────────────────────
def score_resume(resume_text: str, jd_clean: str, vectorizer) -> dict:
    r_clean = clean_text(resume_text)

    # ── A) TF-IDF cosine similarity ──────────────
    tfidf_matrix = vectorizer.transform([r_clean, jd_clean])
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # ── B) Required skill match ──────────────────
    req_found   = extract_skills(resume_text, REQUIRED_SKILLS)
    req_score   = len(req_found) / len(REQUIRED_SKILLS)

    # ── C) Preferred skill match ─────────────────
    pref_found  = extract_skills(resume_text, PREFERRED_SKILLS)
    pref_score  = len(pref_found) / len(PREFERRED_SKILLS)

    # ── D) Weighted composite score ──────────────
    # TF-IDF similarity  : 40%
    # Required skills    : 45%
    # Preferred skills   : 15%
    composite = (0.40 * cos_sim) + (0.45 * req_score) + (0.15 * pref_score)

    # ── E) Missing skills ────────────────────────
    all_found   = extract_skills(resume_text, ALL_SKILLS)
    missing_req  = [s for s in REQUIRED_SKILLS if s not in req_found]
    missing_pref = [s for s in PREFERRED_SKILLS if s not in pref_found]

    return {
        "clean_text"       : r_clean,
        "tfidf_similarity" : round(cos_sim, 4),
        "required_score"   : round(req_score, 4),
        "preferred_score"  : round(pref_score, 4),
        "composite_score"  : round(composite, 4),
        "required_found"   : req_found,
        "preferred_found"  : pref_found,
        "all_skills_found" : all_found,
        "missing_required" : missing_req,
        "missing_preferred": missing_pref,
    }

def assign_fit_label(score: float) -> str:
    if score >= 0.65:
        return "Strong Fit   ✅"
    elif score >= 0.40:
        return "Moderate Fit ⚠️"
    else:
        return "Weak Fit     ❌"

FIT_COLORS = {
    "Strong Fit   ✅": "#22c55e",
    "Moderate Fit ⚠️": "#f59e0b",
    "Weak Fit     ❌": "#ef4444",
}


# ──────────────────────────────────────────────
# 7.  MAIN SCREENING PIPELINE
# ──────────────────────────────────────────────
def run_screening() -> pd.DataFrame:
    print("=" * 62)
    print("   Resume Screening System  |  Role: " + JOB_ROLE)
    print("=" * 62)

    # Fit TF-IDF on all texts
    jd_clean = clean_text(JOB_DESCRIPTION)
    all_texts = [jd_clean] + [clean_text(r["text"]) for r in RESUMES]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    vectorizer.fit(all_texts)

    rows = []
    for r in RESUMES:
        scores = score_resume(r["text"], jd_clean, vectorizer)
        scores["name"] = r["name"]
        rows.append(scores)

    df = pd.DataFrame(rows)
    df["fit_label"] = df["composite_score"].apply(assign_fit_label)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ──────────────────────────────────────────────
# 8.  CONSOLE REPORT
# ──────────────────────────────────────────────
def print_report(df: pd.DataFrame):
    print(f"\n{'─'*62}")
    print(f"  {'Rank':<5} {'Name':<18} {'Score':>6}  {'Fit'}")
    print(f"{'─'*62}")
    for _, row in df.iterrows():
        print(f"  #{row['rank']:<4} {row['name']:<18} {row['composite_score']:>6.1%}  {row['fit_label']}")
    print(f"{'─'*62}\n")

    print("  Score Breakdown (top 3 candidates):")
    print(f"  {'Name':<18} {'TF-IDF':>8} {'Required':>10} {'Preferred':>11}")
    print(f"  {'─'*50}")
    for _, row in df.head(3).iterrows():
        print(f"  {row['name']:<18} {row['tfidf_similarity']:>8.1%} "
              f"{row['required_score']:>10.1%} {row['preferred_score']:>11.1%}")

    print(f"\n{'─'*62}")
    print("  Skill Gap Report — Top 5 Candidates")
    print(f"{'─'*62}")
    for _, row in df.head(5).iterrows():
        print(f"\n  📄 {row['name']}  (Score: {row['composite_score']:.1%}  |  {row['fit_label']})")
        print(f"     ✅ Skills found   : {len(row['all_skills_found'])} / {len(ALL_SKILLS)}")
        if row['missing_required']:
            print(f"     ⛔ Missing Required : {', '.join(row['missing_required'])}")
        else:
            print(f"     ⛔ Missing Required : None — full match!")
        if row['missing_preferred']:
            print(f"     💡 Missing Preferred: {', '.join(row['missing_preferred'][:5])}"
                  + (" …" if len(row['missing_preferred']) > 5 else ""))


# ──────────────────────────────────────────────
# 9.  VISUALISATION DASHBOARD  (redesigned)
# ──────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, filepath: str):
    # ── Palette ──────────────────────────────────
    BG       = "#0d1117"   # page background
    SURFACE  = "#161b22"   # panel surface
    SURFACE2 = "#21262d"   # inner card / alt row
    BORDER   = "#30363d"   # subtle borders
    TEXT1    = "#e6edf3"   # primary text
    TEXT2    = "#8b949e"   # secondary / muted
    BLUE     = "#58a6ff"
    PURPLE   = "#bc8cff"
    TEAL     = "#39d3c3"
    ORANGE   = "#ffa657"
    GREEN    = "#3fb950"
    RED      = "#f85149"
    YELLOW   = "#e3b341"

    FIT_C = {
        "Strong Fit   ✅": GREEN,
        "Moderate Fit ⚠️": YELLOW,
        "Weak Fit     ❌": RED,
    }

    short  = [n.split()[0] for n in df["name"].tolist()]
    scores = df["composite_score"].tolist()
    ranks  = df["rank"].tolist()
    n      = len(df)

    # ── Figure skeleton ─────────────────────────
    fig = plt.figure(figsize=(22, 18), facecolor=BG)

    # Layout: header strip + 2 rows (top row: 3 cols, bottom row: full-width heatmap)
    outer = GridSpec(4, 1, figure=fig,
                     height_ratios=[0.06, 0.38, 0.38, 0.18],
                     hspace=0.44, left=0.05, right=0.97,
                     top=0.95, bottom=0.04)

    # ── HEADER BANNER ───────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_facecolor(SURFACE)
    for sp in ax_hdr.spines.values():
        sp.set_edgecolor(BORDER)
    ax_hdr.set_xticks([]); ax_hdr.set_yticks([])
    ax_hdr.text(0.013, 0.55, "Resume Screening Dashboard",
                transform=ax_hdr.transAxes, color=TEXT1,
                fontsize=17, fontweight="bold", va="center")
    ax_hdr.text(0.013, 0.12, f"Role: {JOB_ROLE}   |   Candidates: {n}   |"
                f"   Required skills: {len(REQUIRED_SKILLS)}   |"
                f"   Preferred skills: {len(PREFERRED_SKILLS)}",
                transform=ax_hdr.transAxes, color=TEXT2,
                fontsize=9, va="center")
    # Mini KPI tiles inside header
    strong = sum(1 for f in df["fit_label"] if "Strong" in f)
    mod    = sum(1 for f in df["fit_label"] if "Moderate" in f)
    weak   = sum(1 for f in df["fit_label"] if "Weak" in f)
    top_score = scores[0]
    kpis = [
        (f"{strong}", "Strong Fits",   GREEN),
        (f"{mod}",    "Moderate Fits", YELLOW),
        (f"{weak}",   "Weak Fits",     RED),
        (f"{top_score:.0%}", "Top Score",   BLUE),
    ]
    for ki, (val, lbl, col) in enumerate(kpis):
        x = 0.60 + ki * 0.10
        ax_hdr.text(x, 0.72, val, transform=ax_hdr.transAxes,
                    color=col, fontsize=14, fontweight="bold",
                    ha="center", va="center")
        ax_hdr.text(x, 0.20, lbl, transform=ax_hdr.transAxes,
                    color=TEXT2, fontsize=7.5, ha="center", va="center")

    # ── TOP ROW: 3 panels ───────────────────────
    top = outer[1].subgridspec(1, 3, wspace=0.35)

    # Panel A: Ranked candidate score bars (horizontal)
    axA = fig.add_subplot(top[0, :2])
    axA.set_facecolor(SURFACE)
    for sp in axA.spines.values(): sp.set_edgecolor(BORDER)
    axA.tick_params(colors=TEXT2, labelsize=8.5)
    axA.xaxis.label.set_color(TEXT2)

    bar_colors = [FIT_C[f] for f in df["fit_label"]]
    y_pos = np.arange(n)
    bars = axA.barh(y_pos, scores, color=bar_colors,
                    height=0.62, edgecolor=BG, linewidth=0.8)
    # Rank badge + name on left
    for i, (name, rank, score) in enumerate(zip(df["name"], ranks, scores)):
        axA.text(-0.005, i, f"#{rank}", ha="right", va="center",
                 color=TEXT2, fontsize=8, fontweight="bold")
        axA.text(score + 0.012, i, f"{score:.1%}",
                 va="center", color=TEXT1, fontsize=8.5, fontweight="bold")

    axA.set_yticks(y_pos)
    axA.set_yticklabels(df["name"].tolist(), fontsize=9, color=TEXT1)
    axA.set_xlim(0, 1.12)
    axA.set_ylim(-0.6, n - 0.4)
    axA.axvline(0.65, color=GREEN,  linestyle="--", lw=1.2, alpha=0.6)
    axA.axvline(0.40, color=YELLOW, linestyle="--", lw=1.2, alpha=0.6)
    axA.set_facecolor(SURFACE)
    axA.grid(axis="x", color=BORDER, linestyle="--", lw=0.5, alpha=0.7)
    axA.set_title("Candidate Ranking by Composite Score",
                  color=TEXT1, fontsize=10.5, fontweight="bold",
                  pad=10, loc="left")
    axA.set_xlabel("Composite Score  (dashed = Strong 65% / Moderate 40% thresholds)",
                   fontsize=8, color=TEXT2)
    axA.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Panel B: Donut — fit distribution
    axB = fig.add_subplot(top[0, 2])
    axB.set_facecolor(SURFACE)
    label_counts = df["fit_label"].value_counts()
    donut_colors = [FIT_C[l] for l in label_counts.index]
    wedges, _, autotexts = axB.pie(
        label_counts.values,
        colors=donut_colors,
        autopct="%1.0f%%",
        startangle=110,
        pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold"); at.set_color(BG)
    axB.text(0, 0, f"{n}\ncandidates", ha="center", va="center",
             color=TEXT1, fontsize=10, fontweight="bold", linespacing=1.5)
    legend_p = [mpatches.Patch(color=FIT_C[l],
                               label=l.split("  ")[0] + f"  ({v})")
                for l, v in zip(label_counts.index, label_counts.values)]
    axB.legend(handles=legend_p, loc="lower center",
               bbox_to_anchor=(0.5, -0.16), fontsize=8,
               framealpha=0, labelcolor=TEXT2, ncol=1)
    axB.set_title("Fit Distribution", color=TEXT1,
                  fontsize=10.5, fontweight="bold", pad=10, loc="left")

    # ── MIDDLE ROW: 3 panels ────────────────────
    mid = outer[2].subgridspec(1, 3, wspace=0.35)

    # Panel C: Grouped bar — score components
    axC = fig.add_subplot(mid[0, :2])
    axC.set_facecolor(SURFACE)
    for sp in axC.spines.values(): sp.set_edgecolor(BORDER)
    axC.tick_params(colors=TEXT2, labelsize=8.5)

    x   = np.arange(n)
    w   = 0.24
    tv  = df["tfidf_similarity"].tolist()
    rv  = df["required_score"].tolist()
    pv  = df["preferred_score"].tolist()

    b1 = axC.bar(x - w,  tv, w, label="TF-IDF similarity",
                 color=BLUE,   alpha=0.90, edgecolor=BG, linewidth=0.5)
    b2 = axC.bar(x,      rv, w, label="Required skills",
                 color=PURPLE, alpha=0.90, edgecolor=BG, linewidth=0.5)
    b3 = axC.bar(x + w,  pv, w, label="Preferred skills",
                 color=TEAL,   alpha=0.90, edgecolor=BG, linewidth=0.5)

    for bars_g, vals in [(b1, tv), (b2, rv), (b3, pv)]:
        for bar, val in zip(bars_g, vals):
            if val > 0.06:
                axC.text(bar.get_x() + bar.get_width() / 2,
                         val + 0.012, f"{val:.0%}",
                         ha="center", va="bottom",
                         color=TEXT2, fontsize=6.8)

    axC.set_xticks(x)
    axC.set_xticklabels(short, fontsize=9, color=TEXT1)
    axC.set_ylim(0, 1.18)
    axC.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axC.grid(axis="y", color=BORDER, linestyle="--", lw=0.5, alpha=0.7)
    axC.set_title("Score Component Breakdown",
                  color=TEXT1, fontsize=10.5, fontweight="bold",
                  pad=10, loc="left")
    axC.legend(fontsize=8, framealpha=0, labelcolor=TEXT2,
               ncol=3, loc="upper right", bbox_to_anchor=(1, 1.02))

    # Panel D: Skills matched — lollipop chart
    axD = fig.add_subplot(mid[0, 2])
    axD.set_facecolor(SURFACE)
    for sp in axD.spines.values(): sp.set_edgecolor(BORDER)
    axD.tick_params(colors=TEXT2, labelsize=8.5)

    skill_counts = df["all_skills_found"].apply(len).tolist()
    total_skills = len(ALL_SKILLS)
    lollipop_colors = [GREEN if c >= total_skills * 0.6
                       else (YELLOW if c >= total_skills * 0.4 else RED)
                       for c in skill_counts]

    for i, (sc, col) in enumerate(zip(skill_counts, lollipop_colors)):
        axD.hlines(i, 0, sc, color=BORDER, linewidth=2, zorder=1)
        axD.scatter(sc, i, color=col, s=80, zorder=3, edgecolors=BG, linewidth=1)
        axD.text(sc + 0.3, i, f"{sc}/{total_skills}",
                 va="center", color=TEXT1, fontsize=8)

    axD.axvline(total_skills, color=RED, linestyle="--",
                lw=1.2, alpha=0.7, label=f"Max ({total_skills})")
    axD.set_yticks(range(n))
    axD.set_yticklabels(short, fontsize=9, color=TEXT1)
    axD.set_xlim(0, total_skills + 4)
    axD.set_ylim(-0.6, n - 0.4)
    axD.grid(axis="x", color=BORDER, linestyle="--", lw=0.5, alpha=0.5)
    axD.set_title("Skills Matched (lollipop)",
                  color=TEXT1, fontsize=10.5, fontweight="bold",
                  pad=10, loc="left")
    axD.legend(fontsize=8, framealpha=0, labelcolor=TEXT2, loc="lower right")

    # ── BOTTOM: Skill heatmap full-width ────────
    bot = outer[3].subgridspec(1, 1)
    axH = fig.add_subplot(bot[0, 0])
    axH.set_facecolor(SURFACE)
    for sp in axH.spines.values(): sp.set_edgecolor(BORDER)

    skill_matrix = np.zeros((n, len(ALL_SKILLS)))
    for i, row in df.iterrows():
        for j, skill in enumerate(ALL_SKILLS):
            skill_matrix[i, j] = 1 if skill in row["all_skills_found"] else 0

    # Custom 2-color map: absent = dark surface, present = teal-green
    from matplotlib.colors import LinearSegmentedColormap
    cmap_custom = LinearSegmentedColormap.from_list(
        "skill", [SURFACE2, "#1f6feb", TEAL], N=256)

    im = axH.imshow(skill_matrix, aspect="auto", cmap=cmap_custom,
                    vmin=0, vmax=1, interpolation="nearest")

    # Skill name labels on x-axis
    axH.set_xticks(range(len(ALL_SKILLS)))
    axH.set_xticklabels(ALL_SKILLS, rotation=40, ha="right",
                        fontsize=7, color=TEXT2)
    axH.set_yticks(range(n))
    axH.set_yticklabels(short, fontsize=9, color=TEXT1)

    # Divider between required / preferred
    div = len(REQUIRED_SKILLS) - 0.5
    axH.axvline(div, color=ORANGE, linewidth=2, linestyle="-", alpha=0.9)

    # Section labels above heatmap
    axH.text(len(REQUIRED_SKILLS) / 2 - 0.5, -0.85,
             "Required Skills", ha="center",
             color=ORANGE, fontsize=8.5, fontweight="bold",
             transform=axH.get_xaxis_transform())
    axH.text(len(REQUIRED_SKILLS) + len(PREFERRED_SKILLS) / 2 - 0.5, -0.85,
             "Preferred Skills", ha="center",
             color=TEAL, fontsize=8.5, fontweight="bold",
             transform=axH.get_xaxis_transform())

    # Cell annotations (tick/cross)
    for i in range(n):
        for j in range(len(ALL_SKILLS)):
            sym   = "✓" if skill_matrix[i, j] == 1 else "·"
            col   = TEXT1 if skill_matrix[i, j] == 1 else BORDER
            fsize = 7.5 if skill_matrix[i, j] == 1 else 8
            axH.text(j, i, sym, ha="center", va="center",
                     color=col, fontsize=fsize)

    axH.set_title("Skill Coverage Heatmap  (✓ = present   · = absent   "
                  "| orange line = required / preferred boundary)",
                  color=TEXT1, fontsize=10.5, fontweight="bold",
                  pad=12, loc="left")

    plt.savefig(filepath, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  [saved] {filepath}")


# ──────────────────────────────────────────────
# 10.  DEMO — PREDICT A SINGLE NEW RESUME
# ──────────────────────────────────────────────
def predict_single(resume_text: str, vectorizer, jd_clean: str) -> None:
    scores = score_resume(resume_text, jd_clean, vectorizer)
    label  = assign_fit_label(scores["composite_score"])
    print("\n" + "─" * 55)
    print("  Single Resume Prediction")
    print("─" * 55)
    print(f"  Composite Score : {scores['composite_score']:.1%}")
    print(f"  Fit Label       : {label}")
    print(f"  TF-IDF Sim      : {scores['tfidf_similarity']:.1%}")
    print(f"  Required Match  : {scores['required_score']:.1%}  "
          f"({len(scores['required_found'])}/{len(REQUIRED_SKILLS)} skills)")
    print(f"  Preferred Match : {scores['preferred_score']:.1%}  "
          f"({len(scores['preferred_found'])}/{len(PREFERRED_SKILLS)} skills)")
    if scores["missing_required"]:
        print(f"  ⛔ Missing Required : {', '.join(scores['missing_required'])}")
    else:
        print("  ⛔ Missing Required : None!")
    print("─" * 55 + "\n")


# ──────────────────────────────────────────────
# 11.  ENTRY POINT
# ──────────────────────────────────────────────
def main():
    # Run full pipeline
    df = run_screening()

    # Console report
    print_report(df)

    # Dashboard
    print("\n[Generating dashboard …]")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "resume_screening_dashboard.png")
    plot_dashboard(df, out_path)

    # Single new resume demo
    new_resume = """
    Data Scientist with 3 years of experience.
    Proficient in Python, pandas, numpy, scikit-learn, machine learning.
    Solid statistics and data analysis skills. SQL for data querying.
    Regression and classification models for retail forecasting.
    Model evaluation and feature engineering experience.
    Used matplotlib for data visualization. Basic TensorFlow knowledge.
    Git and AWS usage. Hypothesis testing familiarity.
    MSc in Data Science.
    """
    # Re-fit vectorizer for demo
    jd_clean  = clean_text(JOB_DESCRIPTION)
    all_texts = [jd_clean] + [clean_text(r["text"]) for r in RESUMES]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    vectorizer.fit(all_texts)
    predict_single(new_resume, vectorizer, jd_clean)

    print(f"\n  Dashboard saved → {out_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()

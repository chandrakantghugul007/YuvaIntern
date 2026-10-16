"""
HYPOTHETICAL case study for the Week 5 evaluation framework.
All data are SYNTHETIC, generated to illustrate how the framework is applied.
They are NOT empirical findings.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

rng = np.random.default_rng(2026)
N = 30  # learners per design

designs = {
    "A: Textbook notes": dict(post_bonus=0,  decay=22, engage=45, appr_gain=0.10),
    "B: Task-based culture": dict(post_bonus=5,  decay=16, engage=68, appr_gain=0.45),
    "C: Thematic project": dict(post_bonus=8,  decay=12, engage=78, appr_gain=0.70),
    "D: Virtual exchange": dict(post_bonus=9,  decay=11, engage=74, appr_gain=0.80),
}

rows = []
for name, p in designs.items():
    pre_vocab = rng.normal(40, 10, N).clip(0, 100)
    post_vocab = (pre_vocab + 20 + p["post_bonus"] + rng.normal(0, 8, N)).clip(0, 100)
    retention = (post_vocab - p["decay"] + rng.normal(0, 7, N)).clip(0, 100)
    engagement = (p["engage"] + rng.normal(0, 12, N)).clip(0, 100)
    appr_pre = rng.normal(3.0, 0.5, N).clip(1, 5)
    appr_post = (appr_pre + p["appr_gain"] + rng.normal(0, 0.4, N)).clip(1, 5)
    for i in range(N):
        rows.append(dict(design=name, learner=i + 1, pre_vocab=pre_vocab[i], post_vocab=post_vocab[i],
                         retention_4wk=retention[i], engagement_pct=engagement[i],
                         appreciation_pre=appr_pre[i], appreciation_post=appr_post[i]))

df = pd.DataFrame(rows)
df.to_csv("hypothetical_case_study_data.csv", index=False)

base = "A: Textbook notes"
def d_cohen(a, b):
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sp

summary = []
for name in designs:
    g = df[df.design == name]
    b = df[df.design == base]
    row = dict(design=name,
               post_vocab_mean=g.post_vocab.mean(), post_vocab_sd=g.post_vocab.std(),
               retention_mean=g.retention_4wk.mean(), retention_sd=g.retention_4wk.std(),
               engagement_mean=g.engagement_pct.mean(),
               appr_change=(g.appreciation_post - g.appreciation_pre).mean(),
               appr_post_mean=g.appreciation_post.mean())
    if name != base:
        t, p = stats.ttest_ind(g.retention_4wk, b.retention_4wk, equal_var=False)
        row["retention_d_vs_A"] = d_cohen(g.retention_4wk, b.retention_4wk)
        row["retention_p_vs_A"] = p
        t2, p2 = stats.ttest_ind(g.engagement_pct, b.engagement_pct, equal_var=False)
        row["engagement_p_vs_A"] = p2
    summary.append(row)
S = pd.DataFrame(summary)
S.to_csv("hypothetical_summary.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(3).to_string())

# Within-design gain (paired) for appreciation
for name in designs:
    g = df[df.design == name]
    t, p = stats.ttest_rel(g.appreciation_post, g.appreciation_pre)
    print(name, "appreciation paired p =", round(p, 4))

# ---------- Charts ----------
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})
short = [n.split(":")[0] for n in designs]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(designs))
w = 0.38
ax.bar(x - w/2, S.post_vocab_mean, w, yerr=S.post_vocab_sd / np.sqrt(N), label="Immediate post-test", color="#7FB3D5", capsize=3)
ax.bar(x + w/2, S.retention_mean, w, yerr=S.retention_sd / np.sqrt(N), label="4-week retention", color="#1F3864", capsize=3)
ax.set_xticks(x, [n.replace(": ", "\n") for n in designs])
ax.set_ylabel("Mean score (0 to 100)")
ax.set_ylim(0, 100)
ax.set_title("Hypothetical: vocabulary and cultural-knowledge retention by design (error bars: SE)")
ax.legend(loc="upper left")
fig.savefig("charts/fig01_retention_by_design.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x, S.engagement_mean, color=sns.color_palette("Set2", 4))
for i, v in enumerate(S.engagement_mean):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_xticks(x, [n.replace(": ", "\n") for n in designs])
ax.set_ylim(0, 100)
ax.set_ylabel("Mean share of learner turns (%)")
ax.set_title("Hypothetical: learner engagement by design")
fig.savefig("charts/fig02_engagement_by_design.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=df.assign(change=df.appreciation_post - df.appreciation_pre), x="design", y="change",
            hue="design", palette="Set2", legend=False, ax=ax)
ax.axhline(0, color="grey", ls="--")
ax.set_xticks(range(len(designs)), [n.replace(": ", "\n") for n in designs])
ax.set_xlabel("")
ax.set_ylabel("Change in cultural appreciation (1 to 5 scale)")
ax.set_title("Hypothetical: change in cultural appreciation, pre to post")
fig.savefig("charts/fig03_appreciation_change.png"); plt.close(fig)
print("charts saved")

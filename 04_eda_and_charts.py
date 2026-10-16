import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# ---------- Load and clean ----------
df = pd.read_csv("logistics_shipments.csv", parse_dates=["ship_date"])
print("Raw shape:", df.shape)
print("Missing:\n", df.isna().sum()[df.isna().sum() > 0])
df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
df["transit_hours"] = df["transit_hours"].fillna(df["transit_hours"].median())
df["total_delivery_hours"] = df["dwell_hours"] + df["transit_hours"]
df["on_time_flag"] = (df["on_time"] == "Yes").astype(int)
df["month"] = df["ship_date"].dt.month
df["weekday"] = df["ship_date"].dt.day_name()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------- Descriptive statistics ----------
num = ["volume_units", "weight_kg", "distance_km", "dwell_hours", "transit_hours",
       "total_delivery_hours", "transport_cost_inr", "handling_cost_inr", "total_cost_inr"]
desc = df[num].describe().T
desc["median"] = df[num].median()
desc["skew"] = df[num].skew()
desc["cv_%"] = df[num].std() / df[num].mean() * 100
print("\nDescriptive statistics:\n", desc[["mean", "median", "std", "min", "max", "skew", "cv_%"]].round(2).to_string())
desc.round(3).to_csv("descriptive_stats.csv")

print("\nOn-time rate overall:", round(df["on_time_flag"].mean() * 100, 1), "%")
print("\nOn-time by mode:\n", (df.groupby("transport_mode")["on_time_flag"].mean() * 100).round(1))
print("\nOn-time by carrier:\n", (df.groupby("carrier")["on_time_flag"].mean() * 100).round(1))
print("\nOn-time by hub:\n", (df.groupby("origin_hub")["on_time_flag"].mean() * 100).round(1))
print("\nMedian delivery hours by mode:\n", df.groupby("transport_mode")["total_delivery_hours"].median().round(1))
print("\nMedian cost by mode:\n", df.groupby("transport_mode")["total_cost_inr"].median().round(0))
print("\nCost share (mean): transport vs handling:",
      round(df["transport_cost_inr"].mean(), 0), round(df["handling_cost_inr"].mean(), 0))

# Correlations
corr = df[num + ["on_time_flag"]].corr(method="pearson")
print("\nCorrelation with total_cost_inr:\n", corr["total_cost_inr"].round(3).sort_values().to_string())
print("\nCorrelation with total_delivery_hours:\n", corr["total_delivery_hours"].round(3).sort_values().to_string())
print("\nCorr distance vs transit_hours (spearman):",
      round(df["distance_km"].corr(df["transit_hours"], method="spearman"), 3))

# Hub x weekday dwell
pivot = df.pivot_table(index="origin_hub", columns="weekday", values="dwell_hours", aggfunc="mean")[weekday_order]
print("\nMean dwell by hub and weekday:\n", pivot.round(1).to_string())
print("\nMumbai Monday vs Mumbai other days mean dwell:",
      round(df[(df.origin_hub == "Mumbai Hub") & (df.weekday == "Monday")]["dwell_hours"].mean(), 1),
      round(df[(df.origin_hub == "Mumbai Hub") & (df.weekday != "Monday")]["dwell_hours"].mean(), 1))

# Monthly
monthly = df.groupby("month").agg(volume=("shipment_id", "count"),
                                  avg_hours=("total_delivery_hours", "mean"),
                                  avg_dwell=("dwell_hours", "mean"),
                                  on_time=("on_time_flag", "mean"))
print("\nMonthly:\n", monthly.round(2).to_string())

# Outliers on delivery hours (IQR)
q1, q3 = df["total_delivery_hours"].quantile([0.25, 0.75])
iqr = q3 - q1
out = df[(df["total_delivery_hours"] < q1 - 1.5 * iqr) | (df["total_delivery_hours"] > q3 + 1.5 * iqr)]
print("\nDelivery-hour outliers (IQR):", len(out), "of", len(df), "bounds",
      round(q1 - 1.5 * iqr, 1), round(q3 + 1.5 * iqr, 1))

# Carrier x mode cost regression slope (INR per km)
slope = np.polyfit(df["distance_km"], df["transport_cost_inr"], 1)[0]
print("\nTransport cost slope INR per km (all modes):", round(slope, 2))
for m in ["Road", "Rail", "Air", "Sea"]:
    s = df[df.transport_mode == m]
    print(m, "slope:", round(np.polyfit(s["distance_km"], s["transport_cost_inr"], 1)[0], 2))


# ---------- Key insight figures (single source for the report) ----------
peak = df["month"] >= 10
print("\nINSIGHTS")
print("peak share of volume %:", round(peak.mean() * 100, 1))
print("on-time peak vs off-peak %:", round(df[peak].on_time_flag.mean() * 100, 1), round(df[~peak].on_time_flag.mean() * 100, 1))
print("hours peak vs off-peak:", round(df[peak].total_delivery_hours.mean(), 1), round(df[~peak].total_delivery_hours.mean(), 1))
print("dwell share of cycle time % all / peak:", round(df.dwell_hours.mean() / df.total_delivery_hours.mean() * 100, 1),
      round(df[peak].dwell_hours.mean() / df[peak].total_delivery_hours.mean() * 100, 1))
print("dwell peak vs off-peak:", round(df[peak].dwell_hours.mean(), 1), round(df[~peak].dwell_hours.mean(), 1))
print("carrier C on-time peak/off-peak %:", round(df[(df.carrier=="Carrier C") & peak].on_time_flag.mean()*100,1), round(df[(df.carrier=="Carrier C") & ~peak].on_time_flag.mean()*100,1))
print("carrier A on-time peak/off-peak %:", round(df[(df.carrier=="Carrier A") & peak].on_time_flag.mean()*100,1), round(df[(df.carrier=="Carrier A") & ~peak].on_time_flag.mean()*100,1))
mm = (df.origin_hub == "Mumbai Hub") & (df.weekday == "Monday")
print("Mumbai Monday shipments:", int(mm.sum()), " share of all shipments %:", round(mm.mean()*100,1))
print("Mumbai Monday mean dwell:", round(df[mm].dwell_hours.mean(),1), " other shipments:", round(df[~mm].dwell_hours.mean(),1))
print("Mumbai Monday on-time %:", round(df[mm].on_time_flag.mean()*100,1), " others %:", round(df[~mm].on_time_flag.mean()*100,1))
print("transport share of total cost %:", round(df.transport_cost_inr.sum()/df.total_cost_inr.sum()*100,1))
print("mean total cost by mode:", df.groupby("transport_mode").total_cost_inr.mean().round(0).to_dict())
print("mean transport cost by mode:", df.groupby("transport_mode").transport_cost_inr.mean().round(0).to_dict())
print("mean cost per km by mode:", (df.groupby("transport_mode").apply(lambda g: (g.transport_cost_inr/g.distance_km).mean())).round(2).to_dict())
print("median total hours by mode:", df.groupby("transport_mode").total_delivery_hours.median().round(1).to_dict())
print("SLA hours by mode:", df.groupby("transport_mode").sla_hours.first().to_dict())
hi = df.total_delivery_hours.quantile(0.75) + 1.5 * (df.total_delivery_hours.quantile(0.75) - df.total_delivery_hours.quantile(0.25))
lo = df.total_delivery_hours.quantile(0.25) - 1.5 * (df.total_delivery_hours.quantile(0.75) - df.total_delivery_hours.quantile(0.25))
outl = df[(df.total_delivery_hours > hi) | (df.total_delivery_hours < lo)]
print("IQR outliers:", len(outl), "pct:", round(len(outl)/len(df)*100,1), "peak share %:", round((outl.month>=10).mean()*100,1),
      "CarrierC share %:", round((outl.carrier=="Carrier C").mean()*100,1), "Mumbai Monday share %:", round(((outl.origin_hub=="Mumbai Hub")&(outl.weekday=="Monday")).mean()*100,1))
print("Carrier C share of volume %:", round((df.carrier=="Carrier C").mean()*100,1))
print("Volume share by hub %:", (df.origin_hub.value_counts(normalize=True)*100).round(1).to_dict())
print("corr weight vs total cost:", round(df.weight_kg.corr(df.total_cost_inr),3))

df.to_csv("clean_logistics.csv", index=False)

# ---------- Charts ----------
palette = sns.color_palette("Set2")

# 1 Distribution of total delivery hours
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(df["total_delivery_hours"], bins=50, kde=True, color=palette[0], ax=ax)
ax.axvline(df["total_delivery_hours"].mean(), color="red", ls="--", label=f"Mean {df.total_delivery_hours.mean():.1f} h")
ax.axvline(df["total_delivery_hours"].median(), color="black", ls=":", label=f"Median {df.total_delivery_hours.median():.1f} h")
ax.set(title="Distribution of total delivery time (dwell + transit)", xlabel="Hours", ylabel="Shipments")
ax.legend()
fig.savefig("charts/fig01_delivery_hours_distribution.png"); plt.close(fig)

# 2 Boxplot by mode
fig, ax = plt.subplots(figsize=(8, 4.5))
order = ["Air", "Road", "Rail", "Sea"]
sns.boxplot(data=df, x="transport_mode", y="total_delivery_hours", order=order, palette="Set2", ax=ax, fliersize=2)
ax.set(title="Delivery time by transport mode", xlabel="Transport mode", ylabel="Total delivery hours")
fig.savefig("charts/fig02_delivery_by_mode_box.png"); plt.close(fig)

# 3 Correlation heatmap
fig, ax = plt.subplots(figsize=(8.5, 7))
cols = ["volume_units", "weight_kg", "distance_km", "dwell_hours", "transit_hours",
        "total_delivery_hours", "transport_cost_inr", "handling_cost_inr", "total_cost_inr", "on_time_flag"]
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": 0.7})
ax.set_title("Correlation matrix of operational and cost variables")
fig.savefig("charts/fig03_correlation_heatmap.png"); plt.close(fig)

# 4 Distance vs transport cost by mode
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="transport_cost_inr",
                hue="transport_mode", hue_order=order, alpha=0.6, s=18, ax=ax, palette="Set2")
ax.set(title="Distance vs transport cost by mode (sample of 1,500)", xlabel="Distance (km)", ylabel="Transport cost (INR)")
ax.legend(title="Mode")
fig.savefig("charts/fig04_distance_vs_cost_scatter.png"); plt.close(fig)

# 5 On-time rate by carrier
ot = (df.groupby("carrier")["on_time_flag"].mean() * 100).reset_index()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.barplot(data=ot, x="carrier", y="on_time_flag", palette="Set2", ax=ax)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=9)
ax.set(title="On-time delivery rate by carrier", xlabel="Carrier", ylabel="On-time rate (%)", ylim=(0, 100))
fig.savefig("charts/fig05_ontime_by_carrier.png"); plt.close(fig)

# 6 Monthly volume and delivery hours (dual axis)
fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.bar(monthly.index, monthly["volume"], color=palette[0], alpha=0.7, label="Shipments")
ax1.set(xlabel="Month (2025)", ylabel="Shipments", xticks=range(1, 13))
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["avg_hours"], color="crimson", marker="o", label="Avg delivery hours")
ax2.set_ylabel("Average delivery hours")
ax2.set_ylim(0, 40)  # zero baseline so the peak rise is not exaggerated
ax1.set_ylim(0, 600)
ax1.set_title("Monthly shipment volume and average delivery time")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=True)
fig.savefig("charts/fig06_monthly_volume_and_hours.png"); plt.close(fig)

# 7 Heatmap dwell by hub x weekday
fig, ax = plt.subplots(figsize=(8.5, 4.5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean dwell (h)"})
ax.set(title="Average warehouse dwell time by hub and weekday", xlabel="Weekday", ylabel="Origin hub")
fig.savefig("charts/fig07_dwell_hub_weekday_heatmap.png"); plt.close(fig)

# 8 Cost composition by mode (stacked)
comp = df.groupby("transport_mode")[["transport_cost_inr", "handling_cost_inr"]].mean().loc[order]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
comp.plot(kind="bar", stacked=True, ax=ax, color=[palette[0], palette[1]], rot=0)
ax.set(title="Average cost composition by transport mode", xlabel="Transport mode", ylabel="Mean cost (INR)")
ax.legend(["Transport", "Handling"])
fig.savefig("charts/fig08_cost_composition_by_mode.png"); plt.close(fig)

print("\nCharts saved:", 8)

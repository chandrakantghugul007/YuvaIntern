import numpy as np
import pandas as pd

# ---- Step 1: Load and inspect ----
raw = pd.read_csv("raw_shipments.csv")
print("Shape:", raw.shape)
print("Missing per column:\n", raw.isna().sum())
print("Exact duplicate rows:", raw.duplicated().sum())
print("Transport modes:\n", raw["Transport Mode"].value_counts())
print("Carriers:\n", raw["Carrier"].value_counts(dropna=False))
print("Raw dtypes:\n", raw.dtypes)
print("Weight describe:\n", raw["Weight KG"].describe())
print("Distance min:", raw["Distance KM"].min())

log = {"raw_rows": len(raw)}
df = raw.copy()

# ---- Step 2: Standardise column names and labels ----
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
text_cols = df.select_dtypes(exclude="number").columns
df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
df["transport_mode"] = df["transport_mode"].str.title()
carrier_map = {"BLUEDART": "BlueDart", "DHL": "DHL", "DELHIVERY": "Delhivery",
               "FEDEX": "FedEx", "XPRESSBEES": "XpressBees"}
df["carrier"] = df["carrier"].str.upper().map(carrier_map)

# ---- Step 3: Type conversion ----
for c in ["ship_date", "delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")

# ---- Step 4: Remove duplicates ----
before = len(df)
df = df.drop_duplicates()
log["duplicates_removed"] = before - len(df)

# ---- Step 5: Invalid values ----
neg = df["distance_km"] <= 0
log["negative_distance_fixed"] = int(neg.sum())
df.loc[neg, "distance_km"] = np.nan

bad_dates = df["delivery_date"] < df["ship_date"]
log["delivery_before_ship"] = int(bad_dates.sum())
df.loc[bad_dates, ["ship_date", "delivery_date"]] = np.nan
before = len(df)
df = df.dropna(subset=["ship_date", "delivery_date"])
log["rows_dropped_bad_dates"] = before - len(df)

# ---- Step 6: Missing values ----
print("\nMissing before imputation:\n", df.isna().sum())
log["missing_before_impute"] = int(df.isna().sum().sum())
df["carrier"] = df["carrier"].fillna("Unknown")
for col in ["weight_kg", "distance_km"]:
    df[col] = df.groupby("transport_mode")[col].transform(lambda s: s.fillna(s.median()))

# ---- Step 7: Outliers (IQR, winsorise, flag) ----
q1, q3 = df["weight_kg"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = max(q1 - 1.5 * iqr, 0.0), q3 + 1.5 * iqr  # weight cannot be negative
df["weight_outlier_flag"] = ~df["weight_kg"].between(low, high)
log["weight_outliers_flagged"] = int(df["weight_outlier_flag"].sum())
log["weight_bounds"] = (round(low, 2), round(high, 2))
df["weight_kg"] = df["weight_kg"].clip(low, high)

# ---- Step 8: Derived features ----
df["transit_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
df["cost_per_kg_inr"] = (df["delivery_cost_inr"] / df["weight_kg"]).round(2)

# ---- Step 9: Normalisation (demo; scaling applied to numeric features) ----
num_cols = ["weight_kg", "distance_km", "transit_days", "delivery_cost_inr"]
minmax = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
zscore = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
df["weight_log"] = np.log1p(df["weight_kg"])
print("\nSkew weight raw vs log:", round(df["weight_kg"].skew(), 3), round(df["weight_log"].skew(), 3))
print("Minmax range:", minmax.min().round(2).tolist(), minmax.max().round(2).tolist())
print("Zscore mean/std:", zscore.mean().round(3).tolist(), zscore.std().round(3).tolist())

# ---- Step 10: Validation ----
assert df["shipment_id"].is_unique
assert (df["distance_km"] > 0).all()
assert (df["transit_days"] >= 0).all()
assert df[num_cols].notna().all().all()
log["clean_rows"] = len(df)

df.to_csv("clean_shipments.csv", index=False)
for k, v in log.items():
    print(k, ":", v)
print("Clean shape:", df.shape)
print(df.describe().round(2).to_string())

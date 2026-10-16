import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

carriers = ["BlueDart", "DHL", "Delhivery", "FedEx", "XpressBees"]
modes = ["Road", "Rail", "Air", "Sea"]
regions = ["North", "South", "East", "West", "Central"]

df = pd.DataFrame({
    "Shipment ID": [f"SHP{i:05d}" for i in range(1, n + 1)],
    "Origin Region": rng.choice(regions, n),
    "Destination Region": rng.choice(regions, n),
    "Transport Mode": rng.choice(modes, n, p=[0.60, 0.15, 0.10, 0.15]),
    "Carrier": rng.choice(carriers, n),
    "Weight KG": rng.lognormal(mean=2.5, sigma=0.6, size=n).round(2),
    "Distance KM": rng.normal(850, 300, n).clip(20).round(1),
    "Ship Date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
})
transit = (df["Distance KM"] / 400 + rng.normal(1.5, 0.8, n)).clip(0.5)
df["Delivery Date"] = df["Ship Date"] + pd.to_timedelta(np.ceil(transit), unit="D")
df["Delivery Cost INR"] = (df["Weight KG"] * 12 + df["Distance KM"] * 1.8 + rng.normal(0, 150, n)).round(2)

# Inject realistic data-quality defects
for col, frac in {"Weight KG": 0.05, "Distance KM": 0.03, "Carrier": 0.04}.items():
    df.loc[df.sample(frac=frac, random_state=1).index, col] = np.nan

idx = df.sample(frac=0.08, random_state=2).index
df.loc[idx, "Transport Mode"] = df.loc[idx, "Transport Mode"].str.lower()

idx = df.sample(frac=0.05, random_state=3).index
df.loc[idx, "Carrier"] = df.loc[idx, "Carrier"].str.upper() + " "

idx = df.sample(5, random_state=4).index
df.loc[idx, "Weight KG"] = df.loc[idx, "Weight KG"] * 100

idx = df.sample(3, random_state=7).index
df.loc[idx, "Distance KM"] = -df.loc[idx, "Distance KM"]

idx = df.sample(4, random_state=6).index
df.loc[idx, "Delivery Date"] = df.loc[idx, "Ship Date"] - pd.Timedelta(days=2)

df = pd.concat([df, df.sample(20, random_state=5)], ignore_index=True)

df.to_csv("raw_shipments.csv", index=False)
print("Raw rows written:", len(df))

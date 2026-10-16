import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 5000

hubs = ["Nagpur Hub", "Mumbai Hub", "Pune Hub", "Delhi Hub", "Chennai Hub"]
hub_p = [0.25, 0.25, 0.2, 0.18, 0.12]
carriers = ["Carrier A", "Carrier B", "Carrier C", "Carrier D"]
modes = ["Road", "Rail", "Air", "Sea"]
mode_p = [0.62, 0.18, 0.08, 0.12]

dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
df = pd.DataFrame({
    "shipment_id": [f"SH{i:06d}" for i in range(1, n + 1)],
    "ship_date": dates,
    "origin_hub": rng.choice(hubs, n, p=hub_p),
    "carrier": rng.choice(carriers, n, p=[0.35, 0.30, 0.20, 0.15]),
    "transport_mode": rng.choice(modes, n, p=mode_p),
    "volume_units": rng.integers(1, 60, n),
    "weight_kg": rng.lognormal(3.0, 0.7, n).round(1),
    "distance_km": rng.normal(600, 260, n).clip(50, 2000).round(0),
})

# Mode speed (km/h effective, including loading) and base cost per km
speed = df["transport_mode"].map({"Road": 45, "Rail": 38, "Air": 350, "Sea": 22})
rate = df["transport_mode"].map({"Road": 7.0, "Rail": 4.5, "Air": 14.0, "Sea": 3.5})

# Warehouse dwell: bottleneck at Mumbai Hub on Mondays and in peak months (Oct-Dec)
dwell = rng.gamma(shape=2.0, scale=5.0, size=n)
dwell += np.where((df["origin_hub"] == "Mumbai Hub") & (df["ship_date"].dt.dayofweek == 0), 14, 0)
dwell += np.where(df["ship_date"].dt.month.isin([10, 11, 12]), 6, 0)
df["dwell_hours"] = dwell.round(1)

carrier_delay = df["carrier"].map({"Carrier A": 0, "Carrier B": 2, "Carrier C": 5, "Carrier D": 1})
transit = df["distance_km"] / speed + rng.normal(0, 4, n) + carrier_delay
df["transit_hours"] = transit.clip(lower=1).round(1)
df["total_delivery_hours"] = (df["dwell_hours"] + df["transit_hours"]).round(1)

# Promised SLA by mode, on-time if total delivery within SLA
sla = df["transport_mode"].map({"Road": 28, "Rail": 32, "Air": 16, "Sea": 44})
df["sla_hours"] = sla
df["on_time"] = np.where(df["total_delivery_hours"] <= df["sla_hours"], "Yes", "No")

# Costs
df["transport_cost_inr"] = (df["distance_km"] * rate + df["weight_kg"] * 1.5
                            + rng.normal(0, 60, n)).clip(lower=50).round(2)
df["handling_cost_inr"] = (df["volume_units"] * 18 + df["dwell_hours"] * 4
                           + rng.normal(0, 25, n)).clip(lower=10).round(2)
df["total_cost_inr"] = (df["transport_cost_inr"] + df["handling_cost_inr"]).round(2)

# Inject a few realistic quality issues (cleaned in analysis)
df.loc[df.sample(40, random_state=1).index, "weight_kg"] = np.nan
df.loc[df.sample(25, random_state=2).index, "transit_hours"] = np.nan

df.to_csv("logistics_shipments.csv", index=False)
print("Rows:", len(df), "Columns:", df.shape[1])
print(df.head(3).T)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

np.random.seed(21)
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. CREATE SYNTHETIC REAL-WORLD-STYLE EV DATA
# ---------------------------------------------------------
stations = [
    ("EV001", "Central Business District", "Fast"),
    ("EV002", "Airport Corridor", "Fast"),
    ("EV003", "Tech Park", "Fast"),
    ("EV004", "Residential Zone", "Standard"),
    ("EV005", "Shopping Mall", "Fast"),
    ("EV006", "University Area", "Standard"),
    ("EV007", "Highway Exit", "Fast"),
    ("EV008", "IT Corridor", "Fast"),
    ("EV009", "Suburban Hub", "Standard"),
    ("EV010", "Metro Station", "Fast"),
    ("EV011", "Industrial Area", "Standard"),
    ("EV012", "Hospital District", "Standard"),
]

hours = pd.date_range("2025-01-01", "2025-06-30 23:00", freq="h")
rows = []

for station_id, location, charger_type in stations:
    base = np.random.randint(10, 26)

    for ts in hours:
        h = ts.hour
        weekday = ts.weekday()

        morning = 1.0 if 7 <= h <= 10 else 0
        evening = 1.25 if 17 <= h <= 21 else 0
        weekend = 0.82 if weekday >= 5 else 1.0

        location_factor = {
            "Central Business District": 1.35,
            "Airport Corridor": 1.25,
            "Tech Park": 1.30,
            "Residential Zone": 0.75,
            "Shopping Mall": 1.15,
            "University Area": 0.70,
            "Highway Exit": 1.20,
            "IT Corridor": 1.28,
            "Suburban Hub": 0.82,
            "Metro Station": 1.18,
            "Industrial Area": 0.72,
            "Hospital District": 0.85,
        }[location]

        demand = base * (1 + morning + evening) * weekend * location_factor
        sessions = max(0, int(np.random.normal(demand, max(2, demand * .18))))

        avg_kwh = np.random.uniform(18, 34)
        energy = max(0, sessions * avg_kwh * np.random.uniform(.88, 1.12))

        price = np.random.uniform(12, 18)
        revenue = energy * price

        utilization = min(
            100,
            sessions / (45 if charger_type == "Fast" else 30) * 100
        )

        rows.append([
            ts, station_id, location, charger_type,
            sessions, round(energy, 2),
            round(price, 2), round(revenue, 2),
            round(utilization, 2)
        ])

df = pd.DataFrame(rows, columns=[
    "timestamp", "station_id", "location", "charger_type",
    "sessions", "energy_kwh", "price_per_kwh",
    "revenue", "utilization_pct"
])

# ---------------------------------------------------------
# 2. KPI ANALYSIS
# ---------------------------------------------------------
total_sessions = df["sessions"].sum()
total_energy = df["energy_kwh"].sum()
total_revenue = df["revenue"].sum()
avg_utilization = df["utilization_pct"].mean()

print("=" * 65)
print("EV CHARGING STATION DEMAND & LOCATION INTELLIGENCE")
print("=" * 65)
print(f"Total charging sessions : {total_sessions:,.0f}")
print(f"Energy delivered        : {total_energy:,.0f} kWh")
print(f"Revenue                 : ₹{total_revenue:,.0f}")
print(f"Average utilization     : {avg_utilization:.1f}%")

# ---------------------------------------------------------
# 3. STATION SCORECARD
# ---------------------------------------------------------
summary = (
    df.groupby(["station_id", "location", "charger_type"])
      .agg(
          sessions=("sessions", "sum"),
          energy_kwh=("energy_kwh", "sum"),
          revenue=("revenue", "sum"),
          avg_utilization=("utilization_pct", "mean")
      )
      .reset_index()
)

summary["revenue_per_session"] = (
    summary["revenue"] / summary["sessions"]
).round(2)

summary["demand_score"] = (
    summary["avg_utilization"] * .55
    + (summary["energy_kwh"] / summary["energy_kwh"].max()) * 100 * .30
    + (summary["sessions"] / summary["sessions"].max()) * 100 * .15
).round(2)

summary["priority"] = pd.cut(
    summary["demand_score"],
    bins=[-1, 35, 65, 101],
    labels=["Monitor", "Growth Opportunity", "Capacity Priority"]
)

summary = summary.sort_values("demand_score", ascending=False)

print("\nTOP STATIONS")
print(summary.head(10).round(2).to_string(index=False))

summary.to_csv("station_summary.csv", index=False)

# ---------------------------------------------------------
# 4. PEAK-HOUR ANALYSIS
# ---------------------------------------------------------
hourly = (
    df.assign(hour=df["timestamp"].dt.hour)
      .groupby("hour")
      .agg(
          sessions=("sessions", "sum"),
          energy_kwh=("energy_kwh", "sum"),
          utilization=("utilization_pct", "mean")
      )
)

peak_hour = hourly["sessions"].idxmax()
print(f"\nPeak charging hour: {peak_hour}:00")
print(f"Peak-hour sessions: {hourly.loc[peak_hour, 'sessions']:,.0f}")

# ---------------------------------------------------------
# 5. UNDER-UTILIZED STATIONS
# ---------------------------------------------------------
under = summary.nsmallest(4, "avg_utilization")

print("\nUNDER-UTILIZED STATIONS")
print(
    under[
        ["station_id", "location", "avg_utilization", "revenue"]
    ].round(2).to_string(index=False)
)

# ---------------------------------------------------------
# 6. CUSTOMER/LOCATION SEGMENTATION WITH K-MEANS
# ---------------------------------------------------------
features = summary[[
    "sessions", "energy_kwh", "avg_utilization"
]]

scaled = (features - features.mean()) / features.std()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
summary["station_cluster"] = kmeans.fit_predict(scaled)

cluster_profile = (
    summary.groupby("station_cluster")
    [["sessions", "energy_kwh", "avg_utilization", "revenue"]]
    .mean()
    .round(2)
)

print("\nSTATION CLUSTERS")
print(cluster_profile)

cluster_profile.to_csv("station_clusters.csv")

# ---------------------------------------------------------
# 7. VISUALIZATIONS
# ---------------------------------------------------------

# Station utilization
plot_df = summary.sort_values("avg_utilization").tail(8)

plt.figure(figsize=(10, 6))
plt.barh(plot_df["location"], plot_df["avg_utilization"])
plt.title("Top EV Station Utilization")
plt.xlabel("Average Utilization (%)")
plt.tight_layout()
plt.savefig(OUT / "station_utilization.png", dpi=180)
plt.close()

# Hourly demand
plt.figure(figsize=(10, 6))
plt.plot(hourly.index, hourly["sessions"], marker="o")
plt.title("EV Charging Demand by Hour")
plt.xlabel("Hour")
plt.ylabel("Charging Sessions")
plt.xticks(range(24))
plt.grid(alpha=.2)
plt.tight_layout()
plt.savefig(OUT / "hourly_demand.png", dpi=180)
plt.close()

# Revenue
rev = summary.sort_values("revenue").tail(8)

plt.figure(figsize=(10, 6))
plt.barh(rev["location"], rev["revenue"])
plt.title("Top Stations by Revenue")
plt.xlabel("Revenue (₹)")
plt.tight_layout()
plt.savefig(OUT / "station_revenue.png", dpi=180)
plt.close()

# Energy vs utilization
plt.figure(figsize=(9, 6))
plt.scatter(
    summary["energy_kwh"],
    summary["avg_utilization"],
    s=90
)
plt.title("Energy Delivered vs Station Utilization")
plt.xlabel("Energy Delivered (kWh)")
plt.ylabel("Average Utilization (%)")
plt.tight_layout()
plt.savefig(OUT / "energy_vs_utilization.png", dpi=180)
plt.close()

# Demand heatmap-style matrix
heat = (
    df.assign(
        day=df["timestamp"].dt.day_name(),
        hour=df["timestamp"].dt.hour
    )
    .pivot_table(
        index="day",
        columns="hour",
        values="sessions",
        aggfunc="sum"
    )
    .reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
    )
)

plt.figure(figsize=(13, 5))
plt.imshow(heat, aspect="auto")
plt.colorbar(label="Sessions")
plt.title("Charging Demand Heatmap")
plt.xlabel("Hour")
plt.ylabel("Day")
plt.xticks(range(24))
plt.yticks(range(7), heat.index)
plt.tight_layout()
plt.savefig(OUT / "demand_heatmap.png", dpi=180)
plt.close()

# ---------------------------------------------------------
# 8. BUSINESS RECOMMENDATIONS
# ---------------------------------------------------------
priority = summary[summary["priority"] == "Capacity Priority"]

recommendations = f"""
EV CHARGING BUSINESS RECOMMENDATIONS

1. Capacity Planning
   {len(priority)} station(s) fall into the Capacity Priority group.
   Review these locations for additional chargers, queue monitoring,
   or charger-capacity upgrades.

2. Peak Demand
   Charging demand peaks around {peak_hour}:00.
   Staffing, maintenance and dynamic pricing can be evaluated around
   high-demand periods.

3. Under-utilization
   Low-utilization stations should be reviewed for location fit,
   visibility, pricing, partnerships and customer acquisition.

4. Revenue Growth
   Prioritize high-utilization stations for cross-selling,
   membership programs and premium fast-charging options.

5. Expansion Strategy
   Expansion should combine utilization, energy delivered,
   revenue and local demand—not revenue alone.
"""

Path("business_recommendations.txt").write_text(
    recommendations.strip(), encoding="utf-8"
)

df.to_csv("ev_charging_data.csv", index=False)

print("\nProject completed.")
print("Dataset: ev_charging_data.csv")
print("Station scorecard: station_summary.csv")
print("Cluster profile: station_clusters.csv")
print("Recommendations: business_recommendations.txt")
print("Charts saved in output/")

"""
Module 1: DATA COLLECTION
-------------------------
Creates a sample dataset of previous construction projects.

NOTE: Real project data is not available, so this script generates a
realistic SYNTHETIC dataset. If you have real data, save it as
data/construction_data.csv with the same column names and skip this step.

Cost unit: Indian Rupees in Lakhs (1 lakh = 100,000 INR).
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1000


def generate(n=N):
    area = RNG.uniform(600, 6000, n)                    # built-up area (sq ft)
    floors = RNG.integers(1, 6, n)                      # number of floors
    material_rate = RNG.uniform(900, 2200, n)           # material cost (INR / sq ft)
    labour_rate = RNG.uniform(500, 1500, n)             # labour cost (INR / worker / day)
    duration = RNG.uniform(4, 30, n)                    # project duration (months)
    location_tier = RNG.integers(1, 4, n)               # 1 = metro, 2 = city, 3 = town
    quality = RNG.integers(1, 4, n)                     # 1 = basic, 2 = standard, 3 = premium

    total_area = area * floors * 0.6 + area * 0.4       # effective built area
    material_cost = total_area * material_rate
    workers = np.clip(total_area / 250, 4, 120)
    labour_cost = workers * labour_rate * duration * 26  # 26 working days / month

    location_factor = {1: 1.20, 2: 1.00, 3: 0.88}
    quality_factor = {1: 0.90, 2: 1.00, 3: 1.25}
    lf = np.array([location_factor[t] for t in location_tier])
    qf = np.array([quality_factor[q] for q in quality])

    overhead = 0.10 * (material_cost + labour_cost) + 15000 * duration
    base = (material_cost + labour_cost + overhead) * lf * qf
    noise = RNG.normal(1.0, 0.04, n)                     # ~4% random variation
    total_cost_lakhs = base * noise / 100000

    return pd.DataFrame({
        "building_area_sqft": area.round(0),
        "num_floors": floors,
        "material_cost_per_sqft": material_rate.round(0),
        "labour_cost_per_day": labour_rate.round(0),
        "project_duration_months": duration.round(1),
        "location_tier": location_tier,
        "quality_grade": quality,
        "total_cost_lakhs": total_cost_lakhs.round(2),
    })


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/construction_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/construction_data.csv")
    print(df.head())

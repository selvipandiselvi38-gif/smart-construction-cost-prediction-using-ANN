"""
Module 4: COST PREDICTION - predict the cost of a NEW construction project.

Usage:
  python predict.py            (interactive prompts)
  python predict.py --demo     (runs a sample project)
"""
import sys
import joblib
import numpy as np

PROMPTS = [
    ("building_area_sqft", "Building area (sq ft)", 1500),
    ("num_floors", "Number of floors (1-5)", 2),
    ("material_cost_per_sqft", "Material cost (INR per sq ft)", 1400),
    ("labour_cost_per_day", "Labour cost (INR per worker per day)", 800),
    ("project_duration_months", "Project duration (months)", 12),
    ("location_tier", "Location tier (1=metro, 2=city, 3=town)", 2),
    ("quality_grade", "Quality grade (1=basic, 2=standard, 3=premium)", 2),
]


def predict(values: dict) -> float:
    bundle = joblib.load("models/ann_model.joblib")
    x = np.array([[values[f] for f in bundle["features"]]], dtype=float)
    y_s = bundle["model"].predict(bundle["x_scaler"].transform(x))
    return float(bundle["y_scaler"].inverse_transform(y_s.reshape(-1, 1))[0, 0])


def main():
    if "--demo" in sys.argv:
        vals = {k: d for k, _, d in PROMPTS}
    else:
        vals = {}
        for key, label, default in PROMPTS:
            raw = input(f"{label} [{default}]: ").strip()
            vals[key] = float(raw) if raw else float(default)
    cost = predict(vals)
    print("\nProject details:", vals)
    print(f"Predicted construction cost: Rs {cost:,.2f} lakhs "
          f"(~ Rs {cost/100:,.2f} crore)")


if __name__ == "__main__":
    main()

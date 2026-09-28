"""
Modules 2-5: DATA PREPROCESSING, ANN MODEL TRAINING, COST PREDICTION, RESULT ANALYSIS

Usage:  python train.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ann_scratch import ANNRegressor

DATA = "data/construction_data.csv"
TARGET = "total_cost_lakhs"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"Model": name, "MAE (lakhs)": mae, "RMSE (lakhs)": rmse,
            "R2": r2, "MAPE (%)": mape}


def main():
    # ---------- Module 1: load collected data ----------
    if not os.path.exists(DATA):
        from generate_data import generate
        os.makedirs("data", exist_ok=True)
        generate().to_csv(DATA, index=False)
    df = pd.read_csv(DATA)
    print(f"Loaded {df.shape[0]} projects, {df.shape[1]-1} input features")

    # ---------- Module 2: preprocessing ----------
    df = df.drop_duplicates().dropna()
    df = df[df[TARGET] > 0]
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values
    feature_names = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(X.values, y, test_size=0.2,
                                              random_state=42)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr)
    X_te_s = x_scaler.transform(X_te)
    y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

    # ---------- Module 3: ANN training ----------
    print("\nTraining ANN from scratch (NumPy)...")
    scratch = ANNRegressor(layer_sizes=(X.shape[1], 16, 8, 1), lr=0.05,
                           epochs=6000)
    scratch.fit(X_tr_s, y_tr_s)

    print("\nTraining scikit-learn MLPRegressor...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                       max_iter=2000, early_stopping=True, random_state=42)
    mlp.fit(X_tr_s, y_tr_s)

    baseline = LinearRegression().fit(X_tr_s, y_tr_s)   # traditional baseline

    # ---------- Module 4: cost prediction on unseen projects ----------
    def to_cost(model, Xs):
        return y_scaler.inverse_transform(model.predict(Xs).reshape(-1, 1)).ravel()

    p_scratch = to_cost(scratch, X_te_s)
    p_mlp = to_cost(mlp, X_te_s)
    p_lin = to_cost(baseline, X_te_s)

    # ---------- Module 5: result analysis ----------
    res = pd.DataFrame([
        metrics("ANN (from scratch, NumPy)", y_te, p_scratch),
        metrics("ANN (scikit-learn MLP)", y_te, p_mlp),
        metrics("Linear Regression (baseline)", y_te, p_lin),
    ]).round(4)
    print("\n=== RESULT ANALYSIS (test set) ===")
    print(res.to_string(index=False))
    res.to_csv("outputs/results.csv", index=False)

    # Plots
    plt.figure(figsize=(7, 4))
    plt.plot(scratch.loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss (log scale)")
    plt.title("Training loss - ANN from scratch")
    plt.tight_layout(); plt.savefig("outputs/loss_curve.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, p, t in zip(ax, [p_scratch, p_mlp],
                       ["ANN from scratch", "scikit-learn MLP"]):
        a.scatter(y_te, p, s=10, alpha=0.6)
        lim = [min(y_te.min(), p.min()), max(y_te.max(), p.max())]
        a.plot(lim, lim, "r--")
        a.set_xlabel("Actual cost (lakhs)"); a.set_ylabel("Predicted cost (lakhs)")
        a.set_title(t)
    plt.tight_layout(); plt.savefig("outputs/actual_vs_predicted.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(res["Model"].str.replace(" (", "\n(", regex=False), res["MAPE (%)"],
            color=["#3b82f6", "#10b981", "#f59e0b"])
    plt.ylabel("MAPE (%) - lower is better")
    plt.title("Model comparison")
    plt.tight_layout(); plt.savefig("outputs/model_comparison.png", dpi=150); plt.close()

    # Save the best-performing ANN (scikit-learn) for the prediction app
    joblib.dump({"model": mlp, "x_scaler": x_scaler, "y_scaler": y_scaler,
                 "features": feature_names}, "models/ann_model.joblib")
    print("\nSaved model to models/ann_model.joblib and plots to outputs/")


if __name__ == "__main__":
    main()

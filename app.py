from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from functools import wraps
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "static" / "reports"

app = Flask(__name__)
app.secret_key = "smart-construction-ann-secret-key"

MODEL_PATH = MODEL_DIR / "ann_model.keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# Demo login credentials
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception:
        return None, None


def predict_cost(form):
    model, preprocessor = load_artifacts()
    if model is None or preprocessor is None:
        raise RuntimeError("ANN model is not trained. Run: python train_model.py")

    row = pd.DataFrame([{
        "building_area": float(form["building_area"]),
        "floors": int(form["floors"]),
        "material_cost": float(form["material_cost"]),
        "labour_cost": float(form["labour_cost"]),
        "project_duration": float(form["project_duration"]),
        "concrete_quantity": float(form["concrete_quantity"]),
        "steel_quantity": float(form["steel_quantity"]),
        "boq_cost": float(form["boq_cost"]),
        "bbs_cost": float(form["bbs_cost"]),
        "location": form["location"]
    }])

    X = preprocessor.transform(row)
    prediction = float(model.predict(X, verbose=0)[0][0])
    return max(0, prediction)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            prediction = predict_cost(request.form)
            inputs = {
                "Building Area": request.form["building_area"],
                "Floors": request.form["floors"],
                "Material Cost": request.form["material_cost"],
                "Labour Cost": request.form["labour_cost"],
                "Project Duration": request.form["project_duration"],
                "Concrete Quantity": request.form["concrete_quantity"],
                "Steel Quantity": request.form["steel_quantity"],
                "BOQ Cost": request.form["boq_cost"],
                "BBS Cost": request.form["bbs_cost"],
                "Location": request.form["location"]
            }
            session["last_prediction"] = prediction
            session["last_inputs"] = inputs
            return render_template(
                "result.html",
                prediction=prediction,
                inputs=inputs,
                generated_at=datetime.now().strftime("%d-%m-%Y %I:%M %p")
            )
        except Exception as exc:
            flash(str(exc), "danger")
    return render_template("predict.html")


@app.route("/dataset")
@login_required
def dataset():
    csv_path = DATA_DIR / "construction_cost_dataset.csv"
    df = pd.read_csv(csv_path)
    return render_template(
        "dataset.html",
        columns=list(df.columns),
        rows=df.head(50).to_dict(orient="records"),
        total=len(df)
    )


@app.route("/download-report")
@login_required
def download_report():
    prediction = session.get("last_prediction")
    inputs = session.get("last_inputs")
    if prediction is None or inputs is None:
        flash("Please make a prediction first.", "warning")
        return redirect(url_for("predict"))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "construction_cost_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SMART CONSTRUCTION COST PREDICTION USING ANN\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Generated: {datetime.now():%d-%m-%Y %I:%M %p}\n\n")
        f.write("PROJECT INPUTS\n")
        f.write("-" * 30 + "\n")
        for key, value in inputs.items():
            f.write(f"{key}: {value}\n")
        f.write("\nPREDICTED CONSTRUCTION COST\n")
        f.write("-" * 30 + "\n")
        f.write(f"Rs. {prediction:,.2f}\n")
        f.write("\nNote: This result is produced by the trained ANN model.\n")
    return send_file(report_path, as_attachment=True, download_name="construction_cost_report.txt")


if __name__ == "__main__":
    app.run(debug=True)

from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "construction_cost_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "building_area", "floors", "material_cost", "labour_cost",
    "project_duration", "concrete_quantity", "steel_quantity",
    "boq_cost", "bbs_cost", "location"
]
target = "total_cost"

X = df[features]
y = df[target]

numeric_features = [
    "building_area", "floors", "material_cost", "labour_cost",
    "project_duration", "concrete_quantity", "steel_quantity",
    "boq_cost", "bbs_cost"
]
categorical_features = ["location"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.20, random_state=42
)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early_stop = EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.20,
    epochs=300,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

pred = model.predict(X_test, verbose=0).ravel()
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

model.save(MODEL_DIR / "ann_model.keras")
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")

with open(MODEL_DIR / "training_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"MAE  : {mae:.2f}\n")
    f.write(f"RMSE : {rmse:.2f}\n")
    f.write(f"R2   : {r2:.4f}\n")

print("\nANN TRAINING COMPLETED")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")
print("Saved:", MODEL_DIR / "ann_model.keras")

# Smart Construction Cost Prediction using ANN

**Student:** Pandiselvi G | **Reg No:** C4S27602
**Guide:** Dr. M. Gokiladevi, MCA., B.Ed., M.Phil., Ph.D.

## Overview
Traditional construction cost estimation is manual, slow and error-prone. This project uses an
Artificial Neural Network (ANN) that learns from previous project data and predicts the total cost
of a new project from: building area, floors, material cost, labour cost, project duration,
location and quality grade. Costs are in INR lakhs.

## Project Modules (same as the presentation)
| Module | File | What it does |
|---|---|---|
| 1. Data Collection | `generate_data.py` | Builds the project dataset (synthetic - replace with real data) |
| 2. Data Preprocessing | `train.py` | Removes duplicates/nulls, train-test split, standard scaling |
| 3. ANN Model Training | `ann_scratch.py`, `train.py` | ANN from scratch (NumPy) + scikit-learn MLP |
| 4. Cost Prediction | `predict.py` | Predicts cost of a new project |
| 5. Result Analysis | `train.py` | MAE, RMSE, R2, MAPE + plots, compared with Linear Regression |

## ANN Algorithm (as in the slides)
1. Data initialization (random weights, He init)
2. Forward propagation
3. Calculate loss (MSE)
4. Backpropagation
5. Update weights (gradient descent)
6. Repeat until convergence (early stop when loss stops improving)

Architecture (from scratch): 7 inputs -> 16 (ReLU) -> 8 (ReLU) -> 1 output (linear).

## How to run
```bash
pip install -r requirements.txt
python generate_data.py     # creates data/construction_data.csv
python train.py             # trains, evaluates, saves model + plots in outputs/
python predict.py           # enter a new project's details
python predict.py --demo    # quick sample prediction
```

## Results (test set, 200 projects)
| Model | MAE (lakhs) | RMSE (lakhs) | R2 | MAPE % |
|---|---|---|---|---|
| ANN (from scratch, NumPy) | 20.93 | 29.37 | 0.985 | 10.10 |
| ANN (scikit-learn MLP) | 18.56 | 26.88 | 0.987 | 9.37 |
| Linear Regression (baseline) | 63.98 | 97.40 | 0.835 | 40.69 |

The ANN reduces error by roughly 70% compared with the linear baseline, because cost depends
non-linearly on the inputs (area x floors x rate, location and quality multipliers).

Plots in `outputs/`: `loss_curve.png`, `actual_vs_predicted.png`, `model_comparison.png`.

## Using real data
Save your data as `data/construction_data.csv` with these columns and re-run `train.py`:
`building_area_sqft, num_floors, material_cost_per_sqft, labour_cost_per_day,
project_duration_months, location_tier, quality_grade, total_cost_lakhs`.
Note: the results above are on synthetic data; state this clearly in your report/viva.

## Future Enhancements
Real-time market price integration, hybrid ANN + optimization algorithms, mobile app,
environmental/risk factors, cloud deployment.

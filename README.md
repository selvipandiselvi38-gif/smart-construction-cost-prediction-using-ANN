# Smart Construction Cost Prediction Using ANN

This project follows the uploaded project document:

- User Login
- Data Collection
- Data Preprocessing
- ANN Model Training
- Construction Cost Prediction
- Result & Report Generation

## Technology

- Python
- Flask
- HTML, CSS, JavaScript
- Artificial Neural Network (ANN)
- CSV dataset
- Pandas, NumPy, TensorFlow/Keras, Scikit-learn, Matplotlib

## Input fields

The prediction page uses project factors described in the project material:
- Building area
- Number of floors
- Material cost
- Labour cost
- Project duration
- Concrete quantity
- Steel quantity
- BOQ cost
- BBS cost
- Location

## Run the project

### 1. Open Command Prompt in this folder

```bash
cd smart_construction_cost_ann_project
```

### 2. Install libraries

```bash
pip install -r requirements.txt
```

### 3. Train the ANN

```bash
python train_model.py
```

This creates:
- model/ann_model.keras
- model/preprocessor.pkl
- model/training_metrics.txt

### 4. Start Flask

```bash
python app.py
```

### 5. Open browser

http://127.0.0.1:5000

### Demo login

Username: `admin`  
Password: `admin123`

## Important

The included CSV is a DEMO dataset created for testing the application flow. For a final academic project, replace it with actual historical construction project data and retrain the ANN.

## Project flow

Login -> Dashboard -> Enter Construction Details -> Preprocessing -> ANN Prediction -> Predicted Cost -> Report

## ANN process

Data initialization -> Forward propagation -> MSE loss -> Backpropagation -> Weight update -> Repeat until convergence

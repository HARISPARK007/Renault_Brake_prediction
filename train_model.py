import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
file_path = "car_dataset.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Drop irrelevant column
df.drop(columns=["Car ID"], inplace=True)
from sklearn.preprocessing import LabelEncoder


# Standardize Maintenance Type values
df["Maintenance Type"] = df["Maintenance Type"].replace({
    "1 time around 55K": "1 time",
    "2 time brake pad change": "2 time",
    "3 time Brake pad change": "3 time",
    "1 time replaced": "1 time",
    "1 time changed": "1 time",
    "1 time changes due to noise": "1 time",
    "1 time changed due to flood warrentry": "1 time",
    "1 time change around 75K": "1 time",
    "1 time change Approx. 60K": "1 time",
    "1 time around 55K": "1 time",
    "1 times brake pad change": "1 time",
    # Add more replacements as needed
})

# Encode Maintenance Type
label_encoder = LabelEncoder()
df["Maintenance Type"] = label_encoder.fit_transform(df["Maintenance Type"])

# Save the label encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")
df["Brake Failure"] = df["Brake Failure"].map({"Yes": 1, "No": 0})

# Split features and target
X = df.drop(columns=["Brake Failure"])
y = df["Brake Failure"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Skip data augmentation and use the resampled data directly
X_augmented = X_resampled
y_augmented = y_resampled

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with hyperparameter tuning grids
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    },
    "Decision Tree": {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 10]
    },
    "LGBM": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [10, 15, 20]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }
}

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LGBM": LGBMClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

best_models = {}

for name, model in models.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    # Save the best model
    filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(best_model, filename)
    print(f"Saved {name} to {filename}")

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")
print("Saved scaler to models/scaler.pkl")

# Save the label encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("Saved label encoder to models/label_encoder.pkl")
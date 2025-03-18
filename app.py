from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models and scaler
models = {
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "XGBoost": joblib.load("models/xgboost_model.pkl"),
    "LGBM": joblib.load("models/lgbm_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl")
}
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()
            
            # Convert form data to DataFrame
            input_data = pd.DataFrame([form_data])
            
            # Step 1: Standardize Maintenance Type input
            input_data["Maintenance Type"] = input_data["Maintenance Type"].replace({
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
            
            # Step 2: Encode the Maintenance Type column
            # Check if the input Maintenance Type is valid
            valid_maintenance_types = label_encoder.classes_
            if input_data["Maintenance Type"].iloc[0] not in valid_maintenance_types:
                # Map unknown values to a default value (e.g., the most common one)
                input_data["Maintenance Type"] = valid_maintenance_types[0]
            else:
                # Transform valid values using the label encoder
                input_data["Maintenance Type"] = label_encoder.transform([input_data["Maintenance Type"].iloc[0]])
            
            # Step 3: Convert all columns to numeric (except Maintenance Type, which is already encoded)
            input_data = input_data.apply(pd.to_numeric, errors='ignore')
            
            # Step 4: Scale the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Step 5: Make predictions using all models
            predictions = {}
            yes_count = 0
            no_count = 0
            
            for model_name, model in models.items():
                # Get prediction probabilities
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data_scaled)[0]  # [P(No), P(Yes)]
                    prediction = "Yes" if probabilities[1] > 0.5 else "No"
                    probability_yes = round(probabilities[1] * 100, 2)  # Convert to percentage
                    probability_no = round(probabilities[0] * 100, 2)   # Convert to percentage
                    predictions[model_name] = {
                        "prediction": prediction,
                        "probability_yes": probability_yes,
                        "probability_no": probability_no
                    }
                else:
                    # For models without predict_proba (e.g., SVM with probability=False)
                    prediction = model.predict(input_data_scaled)[0]
                    predictions[model_name] = {
                        "prediction": "Yes" if prediction == 1 else "No",
                        "probability_yes": "N/A",
                        "probability_no": "N/A"
                    }
                
                # Count Yes/No for voting
                if predictions[model_name]["prediction"] == "Yes":
                    yes_count += 1
                else:
                    no_count += 1
            
            # Step 6: Voting method for final prediction
            final_prediction = "Yes" if yes_count > no_count else "No"
            
            # Step 7: Feature Importance (Explainable AI)
            feature_importance = {}
            if hasattr(models["Random Forest"], "feature_importances_"):
                # Normalize feature importance to sum to 100
                rf_importance = models["Random Forest"].feature_importances_
                rf_importance_normalized = (rf_importance / rf_importance.sum()) * 100
                feature_importance["Random Forest"] = list(zip(input_data.columns, rf_importance_normalized))
            
            if hasattr(models["XGBoost"], "feature_importances_"):
                # Normalize feature importance to sum to 100
                xgb_importance = models["XGBoost"].feature_importances_
                xgb_importance_normalized = (xgb_importance / xgb_importance.sum()) * 100
                feature_importance["XGBoost"] = list(zip(input_data.columns, xgb_importance_normalized))
            
            if hasattr(models["LGBM"], "feature_importances_"):
                # Normalize feature importance to sum to 100
                lgbm_importance = models["LGBM"].feature_importances_
                lgbm_importance_normalized = (lgbm_importance / lgbm_importance.sum()) * 100
                feature_importance["LGBM"] = list(zip(input_data.columns, lgbm_importance_normalized))
            
            return render_template(
                'result.html',
                predictions=predictions,
                final_prediction=final_prediction,
                feature_importance=feature_importance
            )
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
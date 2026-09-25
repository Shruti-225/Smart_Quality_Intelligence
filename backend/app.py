from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS


# --------------------------------------------------
# Flask setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "random_forest_model.pkl"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "selected_features.pkl"
)


# --------------------------------------------------
# Load trained Random Forest model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
selected_features = joblib.load(FEATURES_PATH)


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.route("/")
def home():

    return jsonify({
        "status": "success",
        "message": "Smart Quality Backend is running!"
    })


# --------------------------------------------------
# Dashboard API
# --------------------------------------------------

@app.route("/api/dashboard")
def dashboard():

    try:

        report_path = (
            PROJECT_ROOT
            / "outputs"
            / "reports"
            / "consolidated_quality_analysis.csv"
        )

        df = pd.read_csv(report_path)

        def get_value(metric):
            row = df[df["metric"] == metric]

            if row.empty:
                return None

            return row.iloc[0]["value"]

        # Production metrics
        total_production = int(
            get_value("Total Production Units")
        )

        good_units = int(
            get_value("Good Units")
        )

        defective_units = int(
            get_value("Defective Units")
        )

        defect_rate = float(
            get_value("Defect Rate")
        ) * 100

        quality_loss = float(
            get_value("Estimated Quality Loss")
        )

        potential_savings = float(
            get_value("Potential Savings")
        )

        # ML metrics
        best_model = get_value(
            "Best Model"
        )

        best_model_f1 = float(
            get_value("Best Model F1 Score")
        )

        best_model_roc_auc = float(
            get_value("Best Model ROC-AUC")
        )

        # Vision metrics
        vision_detection_rate = float(
            get_value("Average Detection Rate")
        ) * 100

        difficult_defect = str(
        get_value("Most Difficult Defect")
        ).replace("_", " ").title()

        difficult_defect_rate = float(
            get_value("Difficult Defect Detection Rate")
        ) * 100

        return jsonify({

            # Production
            "total_production": total_production,
            "good_units": good_units,
            "defective_units": defective_units,
            "defect_rate": round(defect_rate, 2),
            "quality_loss": quality_loss,
            "potential_savings": potential_savings,

            # ML
            "best_model": best_model,
            "best_model_f1": round(best_model_f1, 4),
            "best_model_roc_auc": round(
                best_model_roc_auc,
                4
            ),

            # Vision
            "vision_detection_rate": round(
                vision_detection_rate,
                2
            ),
            "difficult_defect": difficult_defect,
            "difficult_defect_rate": round(
                difficult_defect_rate,
                2
            )

        })

    except Exception as e:

        return jsonify({

            "status": "error",
            "message": str(e)

        }), 500

# --------------------------------------------------
# Production Quality Prediction API
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # Create input dataframe
        input_data = pd.DataFrame(
            [data],
            columns=selected_features
        )

        # Convert values to numeric
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Check for missing values
        if input_data.isnull().any().any():

            return jsonify({
                "status": "error",
                "message": "Please provide valid numeric values for all features."
            }), 400

        # Make prediction
        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = int(
            probability >= 0.40
        )

        if prediction == 1:

            result = "DEFECTIVE"

        else:

            result = "GOOD"

        return jsonify({

            "status": "success",

            "prediction": result,

            "prediction_code": prediction,

            "defect_probability": round(
                probability * 100,
                2
            )

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )
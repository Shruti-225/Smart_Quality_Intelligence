
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ---------- CONFIGURATION ----------
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
VISION_DIR = BASE_DIR / "outputs" / "reports" / "vision"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "random_forest_model.pkl"
FEATURES_PATH = BASE_DIR / "outputs" / "models" / "selected_features.pkl"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "production_cleaned.csv"

st.set_page_config(
    page_title="Smart Quality Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- DATA HELPERS ----------
@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    if not FEATURES_PATH.exists():
        return []
    return list(joblib.load(FEATURES_PATH))


def show_chart(filename, caption):
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Chart not found yet: {filename}")


comparison = load_csv(REPORTS_DIR / "model_comparison.csv")
predictions = load_csv(REPORTS_DIR / "production_predictions.csv")
importance = load_csv(REPORTS_DIR / "feature_importance.csv")
root_causes = load_csv(REPORTS_DIR / "root_cause_analysis.csv")
thresholds = load_csv(REPORTS_DIR / "threshold_analysis.csv")
feature_stats = load_csv(REPORTS_DIR / "feature_statistics.csv")
vision_predictions = load_csv(VISION_DIR / "vision_predictions.csv")
vision_summary = load_csv(VISION_DIR / "defect_type_summary.csv")
vision_performance = load_csv(VISION_DIR / "defect_performance.csv")
vision_thresholds = load_csv(VISION_DIR / "vision_threshold_analysis.csv")

model = load_model()
features = load_features()


# ---------- SIDEBAR ----------
st.sidebar.title("🏭 Smart Quality")
st.sidebar.caption("AI-Powered Quality Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Production Analytics",
        "AI Prediction",
        "Vision Inspection",
        "Root Cause Analysis",
        "Cost Analysis",
        "Model Performance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Smart Quality Intelligence | Development")


# ---------- HEADER ----------
st.title("Smart Quality Intelligence")
st.caption(
    "AI-powered production analytics, defect prediction "
    "and quality inspection."
)


# ---------- OVERVIEW ----------
if page == "Overview":
    st.subheader("Quality Overview")

    total_records = len(predictions)
    actual_defects = (
        int(predictions["actual_label"].sum())
        if total_records and "actual_label" in predictions
        else 0
    )
    predicted_defects = (
        int(predictions["production_prediction"].sum())
        if total_records and "production_prediction" in predictions
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluation Records", f"{total_records:,}")
    c2.metric("Actual Defects", f"{actual_defects:,}")
    c3.metric("Predicted Defects", f"{predicted_defects:,}")
    c4.metric("Selected Features", len(features))

    st.subheader("Project Modules")
    a, b = st.columns(2)
    with a:
        st.markdown("#### 📊 Production Analytics")
        st.write("Explore saved production predictions and quality patterns.")
        st.markdown("#### 🤖 AI Defect Prediction")
        st.write("Score a record using the saved Random Forest model.")
        st.markdown("#### 🔎 Root-Cause Analysis")
        st.write("Explore saved feature importance and root-cause reports.")
    with b:
        st.markdown("#### 🖼️ Vision Inspection")
        st.write("View available vision results and upload inspection images.")
        st.markdown("#### 💰 Cost Analysis")
        st.write("Estimate scrap, rework and downtime costs with editable assumptions.")
        st.markdown("#### 📈 Model Performance")
        st.write("Review saved evaluation metrics and charts.")

    if model is None:
        st.warning("Saved Random Forest model was not found.")
    else:
        st.success("Saved Random Forest model loaded.")

    if not vision_predictions.empty:
        st.info(f"Vision prediction records available: {len(vision_predictions)}")
    else:
        st.warning(
            "No vision prediction CSV was found in outputs/vision. "
            "Vision model inference is not connected yet."
        )


# ---------- PRODUCTION ANALYTICS ----------
elif page == "Production Analytics":
    st.subheader("Production Analytics")

    if predictions.empty:
        st.warning("production_predictions.csv is missing or empty.")
    else:
        total = len(predictions)
        actual = predictions["actual_label"].value_counts().sort_index()
        predicted = predictions["production_prediction"].value_counts().sort_index()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Actual quality labels")
            actual_chart = actual.rename(index={0: "Good", 1: "Defective"})
            st.bar_chart(actual_chart)
        with c2:
            st.markdown("#### Predicted quality labels")
            pred_chart = predicted.rename(index={0: "Good", 1: "Defective"})
            st.bar_chart(pred_chart)

        st.metric("Total evaluation records", f"{total:,}")
        st.markdown("#### Saved prediction records")
        st.dataframe(predictions, use_container_width=True, hide_index=True)

        st.download_button(
            "Download production predictions CSV",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="production_predictions.csv",
            mime="text/csv",
        )

        show_chart("quality_distribution.png", "Saved quality distribution")
        show_chart("dashboard.png", "Saved project dashboard")



# ---------- AI PREDICTION ----------
elif page == "AI Prediction":
    st.subheader("AI Defect Prediction")

    st.markdown("### Upload production data")
    st.write(
        "Upload a CSV containing the 30 selected production features. "
        "The dashboard will validate the data and predict defect probabilities."
    )

    uploaded_csv = st.file_uploader(
        "Choose production CSV",
        type=["csv"],
        key="production_csv_upload",
    )

    if model is None or not features:
        st.error("The saved Random Forest model or feature list is unavailable.")

    elif uploaded_csv is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_csv)

            st.markdown("#### Uploaded data preview")
            st.caption(
                f"{len(uploaded_df):,} rows and {len(uploaded_df.columns):,} columns"
            )
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            missing_columns = [
                feature for feature in features
                if feature not in uploaded_df.columns
            ]

            if missing_columns:
                st.error(
                    f"Your CSV is missing {len(missing_columns)} required "
                    "model features."
                )
                st.write("Missing feature names:", missing_columns)
                st.info(
                    "Upload a CSV with the original feature names. "
                    "Extra columns are allowed, but the 30 selected features are required."
                )

            elif uploaded_df.empty:
                st.error("The uploaded CSV contains no data rows.")

            else:
                # Keep the exact feature order used during model training.
                X_upload = uploaded_df[features].copy()

                # Reject non-numeric or missing values rather than
                # silently inventing measurements.
                for column in features:
                    X_upload[column] = pd.to_numeric(
                        X_upload[column], errors="coerce"
                    )

                invalid_counts = X_upload.isna().sum()
                invalid_counts = invalid_counts[invalid_counts > 0]

                if not invalid_counts.empty:
                    st.error(
                        "Some selected features contain missing or non-numeric "
                        "values. Correct these values in your CSV before prediction."
                    )
                    st.dataframe(
                        invalid_counts.rename("Invalid values").to_frame(),
                        use_container_width=True,
                    )

                elif not X_upload.apply(
                    lambda col: col.map(lambda value: abs(value) != float("inf")).all()
                ).all():
                    st.error("The CSV contains infinite values. Please correct them.")

                else:
                    st.success("CSV validation passed.")

                    if st.button("Predict all uploaded records", type="primary"):
                        try:
                            probabilities = model.predict_proba(X_upload)[:, 1]
                            threshold = 0.40

                            results = uploaded_df.copy()
                            results["defect_probability"] = probabilities
                            results["predicted_label"] = (
                                probabilities >= threshold
                            ).astype(int)
                            results["predicted_quality"] = results[
                                "predicted_label"
                            ].map({0: "Good", 1: "Defective"})

                            good_count = int(
                                (results["predicted_label"] == 0).sum()
                            )
                            defective_count = int(
                                (results["predicted_label"] == 1).sum()
                            )

                            st.markdown("#### Prediction results")
                            a, b, c = st.columns(3)
                            a.metric("Records processed", f"{len(results):,}")
                            b.metric("Predicted good", f"{good_count:,}")
                            c.metric("Predicted defective", f"{defective_count:,}")

                            st.caption(
                                "Predictions use a 0.40 probability threshold. "
                                "These are model estimates, not guaranteed outcomes."
                            )

                            st.dataframe(
                                results,
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.bar_chart(
                                results["predicted_quality"].value_counts()
                            )

                            st.download_button(
                                "Download prediction results CSV",
                                data=results.to_csv(index=False).encode("utf-8"),
                                file_name="production_batch_predictions.csv",
                                mime="text/csv",
                            )

                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")
                            st.info(
                                "Check that the uploaded data uses the same "
                                "units and preprocessing as the training data."
                            )

        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")

    st.divider()
    st.markdown("### Individual record prediction")

    st.write(
        "Alternatively, enter values manually. Use actual production measurements, "
        "not the default zero placeholders."
    )

    if model is not None and features:
        if len(features) != getattr(model, "n_features_in_", len(features)):
            st.error("The saved model and selected feature list do not match.")
        else:
            with st.form("prediction_form"):
                values = {}
                cols = st.columns(3)

                for i, feature in enumerate(features):
                    with cols[i % 3]:
                        values[feature] = st.number_input(
                            f"Feature {feature}",
                            value=0.0,
                            format="%.6f",
                            key=f"manual_input_{feature}",
                        )

                submitted = st.form_submit_button("Predict Individual Record")

            if submitted:
                input_df = pd.DataFrame(
                    [[values[f] for f in features]],
                    columns=features,
                )

                probability = float(model.predict_proba(input_df)[0, 1])
                predicted_label = (
                    "Defective" if probability >= 0.40 else "Good"
                )

                st.metric("Defect probability", f"{probability:.1%}")
                st.metric("Predicted quality", predicted_label)
                st.caption(
                    "This result uses the saved Random Forest and a 0.40 threshold. "
                    "Prediction reliability depends on valid, correctly scaled inputs."
                )




# ---------- VISION INSPECTION ----------
elif page == "Vision Inspection":
    st.subheader("Computer Vision Inspection")
    st.caption("Image anomaly detection results from the trained Isolation Forest pipeline.")

    
    if not vision_predictions.empty:
        total_images = len(vision_predictions)

        # Convert both actual and predicted labels into binary values
        label_map = {
            "good": 0,
            "defective": 1,
            "0": 0,
            "1": 1
        }

        actual_binary = (
            vision_predictions["actual_label"]
            .astype(str).str.strip().str.lower()
            .map(label_map)
        )

        predicted_binary = (
            vision_predictions["predicted_label"]
            .astype(str).str.strip().str.lower()
            .map(label_map)
        )

        # Validate labels before calculating metrics
        if actual_binary.isna().any() or predicted_binary.isna().any():
            st.error("Unknown labels found in vision prediction data.")
            st.write("Actual labels:", vision_predictions["actual_label"].unique())
            st.write("Predicted labels:", vision_predictions["predicted_label"].unique())

        else:
            actual_defects = int((actual_binary == 1).sum())
            detected_defects = int((predicted_binary == 1).sum())

            correct_predictions = int(
                (actual_binary == predicted_binary).sum()
            )

            accuracy = correct_predictions / total_images if total_images else 0

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Images Evaluated", f"{total_images:,}")
            c2.metric("Actual Defects", f"{actual_defects:,}")
            c3.metric("Predicted Defects", f"{detected_defects:,}")
            c4.metric("Evaluation Accuracy", f"{accuracy:.1%}")

            st.divider()

            st.markdown("### Inspection Distribution")

            chart_data = predicted_binary.map({
                0: "Good",
                1: "Defective"
            }).value_counts()

            st.bar_chart(chart_data)


        st.markdown("### Category-wise Inspection")

        if not vision_performance.empty:
            st.dataframe(
                vision_performance,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Category performance report is unavailable.")

        if not vision_summary.empty:
            st.markdown("### Defect Category Distribution")
            st.bar_chart(
                vision_summary.set_index("defect_type")["image_count"]
            )

        st.markdown("### Anomaly Score Analysis")

        if "anomaly_score" in vision_predictions.columns:
            st.line_chart(
                vision_predictions["anomaly_score"].reset_index(drop=True)
            )

        st.markdown("### Inspection Records")

        filter_option = st.selectbox(
            "Filter inspection results",
            ["All", "Good", "Defective"]
        )

        filtered = vision_predictions.copy()

        if filter_option == "Good":
            filtered = filtered[filtered["predicted_label"] == 0]
        elif filter_option == "Defective":
            filtered = filtered[filtered["predicted_label"] == 1]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Vision Inspection Report",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="vision_inspection_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("Vision prediction report is unavailable.")

    if not vision_thresholds.empty:
        with st.expander("View Threshold Analysis"):
            st.dataframe(
                vision_thresholds,
                use_container_width=True,
                hide_index=True
            )

    show_chart(
        "vision_detection_rate.png",
        "Vision detection-rate chart"
    )



# ---------- ROOT CAUSE ANALYSIS ----------
elif page == "Root Cause Analysis":
    st.subheader("Root-Cause Analysis")

    if not root_causes.empty:
        st.markdown("#### Largest reported feature differences")
        st.dataframe(root_causes.head(20), use_container_width=True, hide_index=True)

        if {"feature", "difference"}.issubset(root_causes.columns):
            chart_data = root_causes.head(15).copy()
            chart_data["feature"] = chart_data["feature"].astype(str)
            chart_data = chart_data.set_index("feature")["difference"]
            st.markdown("#### Feature mean differences")
            st.bar_chart(chart_data)
    else:
        st.warning("root_cause_analysis.csv is missing or empty.")

    if not importance.empty:
        st.markdown("#### Feature importance report")
        st.dataframe(importance.head(30), use_container_width=True, hide_index=True)

    show_chart("feature_importance.png", "Saved feature-importance chart")
    show_chart("top_root_causes.png", "Saved top-root-causes chart")


# ---------- COST ANALYSIS ----------
elif page == "Cost Analysis":
    st.subheader("Cost of Poor Quality")
    st.write(
        "Enter your own estimated costs. These assumptions are not measured "
        "factory costs and should be replaced with real business data."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        scrap_cost = st.number_input("Cost per scrapped unit (₹)", min_value=0.0, value=500.0, step=50.0)
    with c2:
        rework_cost = st.number_input("Cost per reworked unit (₹)", min_value=0.0, value=150.0, step=25.0)
    with c3:
        downtime_cost = st.number_input("Downtime cost per hour (₹)", min_value=0.0, value=1000.0, step=100.0)

    c4, c5, c6 = st.columns(3)
    with c4:
        scrap_units = st.number_input("Scrapped units", min_value=0, value=0, step=1)
    with c5:
        rework_units = st.number_input("Reworked units", min_value=0, value=0, step=1)
    with c6:
        downtime_hours = st.number_input("Downtime hours", min_value=0.0, value=0.0, step=0.5)

    scrap_total = scrap_units * scrap_cost
    rework_total = rework_units * rework_cost
    downtime_total = downtime_hours * downtime_cost
    total_cost = scrap_total + rework_total + downtime_total

    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Scrap cost", f"₹{scrap_total:,.2f}")
    b.metric("Rework cost", f"₹{rework_total:,.2f}")
    c.metric("Downtime cost", f"₹{downtime_total:,.2f}")
    d.metric("Total estimated cost", f"₹{total_cost:,.2f}")

    cost_data = pd.DataFrame({
        "Category": ["Scrap", "Rework", "Downtime"],
        "Estimated cost (₹)": [scrap_total, rework_total, downtime_total],
    })
    st.bar_chart(cost_data.set_index("Category"))


# ---------- MODEL PERFORMANCE ----------
elif page == "Model Performance":
    st.subheader("Model Performance")

    if not comparison.empty:
        st.caption(
            "The saved model comparison report uses the evaluation predictions "
            "from model training. The saved production-prediction CSV uses a 0.40 threshold."
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        if "model" in comparison.columns:
            metric_columns = [
                col for col in ["precision", "recall", "f1_score", "roc_auc"]
                if col in comparison.columns
            ]
            for metric in metric_columns:
                st.markdown(f"#### {metric.replace('_', ' ').title()}")
                st.bar_chart(comparison.set_index("model")[[metric]])

        if "model" in comparison.columns and "recall" in comparison.columns:
            rf_rows = comparison[
                comparison["model"].astype(str).str.contains("Random Forest", case=False)
            ]
            if not rf_rows.empty and float(rf_rows.iloc[0]["recall"]) < 0.5:
                st.warning(
                    "The saved Random Forest recall is low. It misses many defective "
                    "records at the evaluated classification setting; investigate threshold "
                    "selection and class imbalance before operational use."
                )
    else:
        st.warning("model_comparison.csv is missing or empty.")

    if not thresholds.empty:
        st.markdown("#### Threshold analysis")
        st.dataframe(thresholds, use_container_width=True, hide_index=True)

    show_chart("confusion_matrix.png", "Saved confusion matrix")
    show_chart("roc_curve.png", "Saved ROC curve")
    show_chart("precision_recall_curve.png", "Saved precision-recall curve")
    show_chart("model_comparison.png", "Saved model comparison")


st.divider()
st.caption("Smart Quality Intelligence | AI-Powered Quality Control")

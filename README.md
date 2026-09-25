# Smart Quality Intelligence

## AI-Powered Manufacturing Quality Control & Defect Prediction

Smart Quality Intelligence is an AI-powered manufacturing quality control system that combines **production-process data, machine learning, computer vision, root-cause analysis, and business analytics** to identify quality issues and support data-driven manufacturing decisions.

The system analyzes production data to predict product quality, analyzes product images for visual defects, identifies process features associated with quality problems, and presents the results through an interactive web dashboard and Power BI analytics.

---

## 🚀 Key Features

- Production quality monitoring
- Machine learning-based defect prediction
- Random Forest quality prediction
- 30 selected production-process features
- Defect probability estimation
- Low / Medium / High risk classification
- Computer vision-based defect inspection
- Visual defect category analysis
- Root-cause analysis
- Business impact estimation
- Quality loss and potential savings analysis
- Interactive Flask web dashboard
- Power BI analytics dashboard

---

## 🏭 Project Workflow

Production Data  
↓  
Data Preprocessing  
↓  
Feature Selection  
↓  
Machine Learning  
↓  
Quality Prediction  
↓  
Computer Vision Inspection  
↓  
Root Cause Analysis  
↓  
Business Impact Analysis  
↓  
Web Dashboard + Power BI

---

## 📊 Production Dataset

The production dataset contains:

- **1,567 production records**
- **1,463 good units**
- **104 defective units**
- **6.64% production defect rate**
- **448 features after preprocessing**
- **30 selected features used for modeling**

---

## 🤖 Machine Learning

Two production-quality classification models were evaluated:

### Logistic Regression

- ROC-AUC: **0.5857**
- F1 Score: **0.1391**
- Defect Recall: **38.10%**

### Random Forest

- ROC-AUC: **0.7605**
- F1 Score: **0.1667**
- Defect Recall: **9.52%**

Among the tested production models, Random Forest achieved the higher ROC-AUC and F1 score. However, its defect recall remains limited, indicating that further model tuning and additional defect-focused data could improve defective-unit detection.

---

## 👁️ Computer Vision

The computer vision component analyzes product images to identify visual anomalies and defect categories.

### Vision Results

| Defect Category | Detection Rate |
|---|---:|
| Good | 100% |
| Manipulated Front | 100% |
| Scratch Head | 100% |
| Scratch Neck | 100% |
| Thread Top | 100% |
| Thread Side | 95.65% |

Overall vision detection rate:

**99.28%**

---

## 💰 Business Impact

The system estimates the financial impact of defective production.

| Metric | Value |
|---|---:|
| Estimated Quality Loss | ₹52,000 |
| Potential Savings | ₹13,000 |
| Estimated Defect Reduction | ~26 units |

The potential savings estimate represents the projected savings from reducing approximately 26 defective units.

---

## 🌐 Web Application

The project includes a Flask backend and an interactive frontend.

### Dashboard

Provides:

- Production KPIs
- Defect rate
- Quality loss
- Potential savings
- ML performance
- Computer vision performance
- AI workflow overview

### Quality Prediction

Allows users to enter the selected production-process features and receive:

- Quality status
- Defect probability
- Risk level

### Analytics

Provides detailed:

- Production analysis
- ML model comparison
- Computer vision performance
- Business impact
- Model performance insights

---

## 🛠️ Technology Stack

### Programming

- Python
- JavaScript
- HTML
- CSS

### Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Logistic Regression

### Computer Vision

- OpenCV
- Image preprocessing
- Anomaly detection

### Backend

- Flask
- Flask-CORS
- Joblib

### Analytics

- Power BI
- Matplotlib

### Development

- VS Code
- Git
- GitHub

---

## 📁 Project Structure

```text
Smart_Quality_Intelligence/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── prediction.html
│   ├── analytics.html
│   ├── script.js
│   └── style.css
│
├── src/
│   ├── analysis.py
│   ├── business_impact.py
│   ├── computer_vision.py
│   ├── data_fusion.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── preprocessing.py
│   ├── production_predictions.py
│   ├── root_cause.py
│   └── visualization.py
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── requirements.txt
└── README.md
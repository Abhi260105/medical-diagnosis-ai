🏥 Medical Diagnosis AI – Multi-Disease Prediction System
An advanced end-to-end machine learning application suite designed to predict the likelihood of Diabetes and Heart Disease in patients based on clinical features. This project includes complete preprocessing, multiple ML models, automated evaluation, comprehensive visualizations, and saved models ready for deployment.

📌 Project Overview
This project is a production-ready AI system for early disease risk detection across multiple conditions. It uses supervised machine learning algorithms trained on structured medical data to classify patient health status.
🩺 Supported Diagnoses:

Diabetes Prediction System - Early detection of diabetes risk
Heart Disease Prediction System - Cardiovascular disease risk assessment

The pipeline includes:

✅ Data cleaning and preprocessing
✅ Exploratory analysis and visualizations
✅ Multiple ML model training and comparison
✅ Ensemble learning techniques
✅ Hyperparameter tuning
✅ Comprehensive evaluation with detailed metrics
✅ Risk stratification and patient reports
✅ Exported trained models
✅ Prediction interface (CLI and API ready)
✅ Batch prediction capabilities
✅ Interactive prediction mode

This repository is suitable for academic projects, research work, healthcare informatics, and real-world ML deployment prototypes.

🚀 Features
🎯 Core Features (Both Systems)
✔ Data Management

Data loading, validation, and preprocessing
Scaling, normalization, and outlier handling
Missing value imputation
Feature engineering

✔ Multiple Machine Learning Algorithms

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
Gradient Boosting
K-Nearest Neighbors
Naive Bayes
Ensemble Voting Classifier

✔ Advanced Model Optimization

Automated hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
Train/test split with stratification
Cross-validation (5-fold StratifiedKFold)
Model performance comparison

✔ Comprehensive Evaluation Metrics

Accuracy
Precision
Recall (Sensitivity)
Specificity
F1-score
Confusion matrix
ROC-AUC score
Classification report
Feature importance analysis

✔ Prediction Capabilities

Single patient prediction
Batch prediction for multiple patients
Interactive CLI prediction mode
Risk level stratification (Low/Moderate/High/Very High)
Detailed patient health reports

✔ Model Persistence

Trained model saving/loading with pickle/joblib
Scaler and metadata preservation
Quick deployment capability

✔ Visualization Suite

Feature distributions
Correlation heatmap
ROC curve
Precision-Recall curve
Confusion matrix
Feature importance plots
Model comparison charts
Prediction probability distributions


📊 System-Specific Features
💉 Diabetes Prediction System (diabetes_predictor.py)

Dataset Features: Glucose, BMI, Age, Blood Pressure, Insulin, etc.
Target: Binary classification (Diabetic / Non-Diabetic)
Model Performance: ~85-90% accuracy
Special Features:

Glucose level analysis
BMI categorization
Age-based risk assessment



❤️ Heart Disease Prediction System (heart_disease_predictor.py)
Dataset Features: Age, Sex, Chest Pain Type, Cholesterol, Max HR, ST Depression, etc.
Target: Binary classification (Heart Disease / Healthy)
Model Performance: ~85-90% accuracy, AUC-ROC ~0.90-0.93
Special Features:

11 clinical parameters
Risk factor identification
Cardiovascular health assessment
Exercise-induced symptoms analysis


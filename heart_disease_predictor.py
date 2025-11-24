"""
Medical Diagnosis AI - Heart Disease Prediction Module
Author: Advanced Medical AI
Date: November 2025
Description: Enhanced machine learning model for heart disease prediction with advanced features
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')


class HeartDiseasePredictor:
    """
    Advanced heart disease prediction system with multiple ML algorithms.
    
    New Features:
    - Multiple model comparison
    - Ensemble learning
    - Risk stratification
    - Batch predictions
    - Model performance comparison
    - Interactive predictions
    - Detailed patient reports
    """
    
    def __init__(self):
        """Initialize the predictor with default parameters."""
        self.model = None
        self.models_dict = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
            'Oldpeak', 'ST_Slope'
        ]
        self.is_trained = False
        self.model_performance = {}
        
    def load_data(self, filepath='data/heart_disease.csv'):
        """
        Load heart disease dataset from CSV file.
        
        Args:
            filepath (str): Path to the heart disease dataset
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            if os.path.exists(filepath):
                self.df = pd.read_csv(filepath)
                print(f"✅ Dataset loaded successfully: {self.df.shape[0]} samples, {self.df.shape[1]} features")
            else:
                print(f"⚠️  File not found at {filepath}")
                print("📥 Creating sample dataset...")
                self.df = self._create_sample_dataset()
            return self.df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("📥 Creating sample dataset...")
            self.df = self._create_sample_dataset()
            return self.df
    
    def _create_sample_dataset(self):
        """Create a sample heart disease dataset."""
        print("🔄 Generating synthetic heart disease dataset...")
        np.random.seed(42)
        n_samples = 300
        
        data = []
        for i in range(n_samples):
            # Determine if patient has heart disease (40% positive cases)
            has_disease = np.random.random() < 0.40
            
            if has_disease:
                # Heart disease patient characteristics
                age = int(np.random.normal(58, 10))
                sex = np.random.choice([0, 1], p=[0.3, 0.7])  # More males
                chest_pain = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.3, 0.3])
                resting_bp = int(np.random.normal(145, 20))
                cholesterol = int(np.random.normal(250, 50))
                fasting_bs = np.random.choice([0, 1], p=[0.6, 0.4])
                resting_ecg = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
                max_hr = int(np.random.normal(130, 20))
                exercise_angina = np.random.choice([0, 1], p=[0.4, 0.6])
                oldpeak = round(np.random.uniform(1.0, 3.0), 1)
                st_slope = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
            else:
                # Healthy patient characteristics
                age = int(np.random.normal(45, 12))
                sex = np.random.choice([0, 1], p=[0.5, 0.5])
                chest_pain = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
                resting_bp = int(np.random.normal(125, 15))
                cholesterol = int(np.random.normal(210, 40))
                fasting_bs = np.random.choice([0, 1], p=[0.8, 0.2])
                resting_ecg = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
                max_hr = int(np.random.normal(160, 15))
                exercise_angina = np.random.choice([0, 1], p=[0.8, 0.2])
                oldpeak = round(np.random.uniform(0.0, 1.5), 1)
                st_slope = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
            
            # Ensure realistic ranges
            age = max(29, min(80, age))
            resting_bp = max(90, min(200, resting_bp))
            cholesterol = max(100, min(600, cholesterol))
            max_hr = max(60, min(220, max_hr))
            oldpeak = max(0.0, min(6.2, oldpeak))
            
            data.append([age, sex, chest_pain, resting_bp, cholesterol, 
                        fasting_bs, resting_ecg, max_hr, exercise_angina, 
                        oldpeak, st_slope, 1 if has_disease else 0])
        
        df = pd.DataFrame(data, columns=[
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
            'Oldpeak', 'ST_Slope', 'HeartDisease'
        ])
        
        print(f"✅ Generated {len(df)} samples")
        return df
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*70)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        print(f"\n1️⃣ Dataset Shape: {self.df.shape}")
        print(f"   Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n2️⃣ Feature Data Types:")
        print(self.df.dtypes)
        
        print("\n3️⃣ Statistical Summary:")
        print(self.df.describe().round(2))
        
        print("\n4️⃣ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   ✅ No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\n5️⃣ Class Distribution:")
        counts = self.df['HeartDisease'].value_counts()
        print(f"   Heart Disease (1): {counts.get(1, 0)} ({counts.get(1, 0)/len(self.df)*100:.2f}%)")
        print(f"   Healthy (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(self.df)*100:.2f}%)")
        
        print("\n6️⃣ Feature Correlations with Target:")
        correlations = self.df.corr()['HeartDisease'].sort_values(ascending=False)
        print(correlations[1:])  # Exclude HeartDisease itself
        
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        print("\n" + "="*70)
        print("🔧 DATA PREPROCESSING")
        print("="*70)
        
        # Check for missing values
        if self.df.isnull().sum().sum() > 0:
            print("⚠️  Handling missing values...")
            self.df.fillna(self.df.median(), inplace=True)
            print("✅ Missing values handled")
        
        # Separate features and target
        X = self.df.drop('HeartDisease', axis=1)
        y = self.df['HeartDisease']
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("✅ Features scaled using StandardScaler")
        
        print(f"✅ Feature names: {self.feature_names}")
        
    def train_multiple_models(self):
        """Train and compare multiple ML models."""
        print("\n" + "="*70)
        print("🚀 TRAINING MULTIPLE MODELS")
        print("="*70)
        
        # Define multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42, kernel='rbf'),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='f1')
            
            # Test predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.models_dict[name] = model
            self.model_performance[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Select best model based on F1 score
        best_model_name = max(self.model_performance, 
                             key=lambda x: self.model_performance[x]['f1_score'])
        self.model = self.models_dict[best_model_name]
        self.best_model_name = best_model_name
        self.is_trained = True
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1 Score: {self.model_performance[best_model_name]['f1_score']:.4f}")
        
    def create_ensemble_model(self):
        """Create an ensemble model combining multiple classifiers."""
        print("\n" + "="*70)
        print("🎯 CREATING ENSEMBLE MODEL")
        print("="*70)
        
        # Create ensemble with best performing models
        ensemble = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ],
            voting='soft'
        )
        
        print("🔄 Training ensemble model...")
        ensemble.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(self.X_test_scaled)
        y_pred_proba = ensemble.predict_proba(self.X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models_dict['Ensemble'] = ensemble
        self.model_performance['Ensemble'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc
        }
        
        print(f"✅ Ensemble Model Performance:")
        print(f"   Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Use ensemble as main model if it performs better
        if f1 > self.model_performance[self.best_model_name]['f1_score']:
            self.model = ensemble
            self.best_model_name = 'Ensemble'
            print(f"🏆 Ensemble model selected as best performer!")
        
    def compare_models(self):
        """Display comprehensive model comparison."""
        print("\n" + "="*70)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.model_performance).T
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        print("\n" + comparison_df.to_string())
        
        # Highlight best model
        print(f"\n🏆 Best Model: {self.best_model_name}")
        
    def evaluate_model(self):
        """Evaluate the trained model with detailed metrics."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return None
        
        print("\n" + "="*70)
        print(f"📈 DETAILED MODEL EVALUATION - {self.best_model_name}")
        print("="*70)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ Precision: {precision:.4f} - Out of predicted positive, {precision*100:.1f}% are correct")
        print(f"✅ Recall:    {recall:.4f} - Detected {recall*100:.1f}% of actual heart disease cases")
        print(f"✅ F1-Score:  {f1:.4f} - Harmonic mean of precision and recall")
        print(f"✅ AUC-ROC:   {auc_roc:.4f} - Model discrimination ability")
        
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Healthy', 'Heart Disease']))
        
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        # Calculate additional metrics
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])
        print(f"\n✅ Specificity: {specificity:.4f} - Correctly identified {specificity*100:.1f}% of healthy patients")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'specificity': specificity,
            'confusion_matrix': cm
        }
    
    def plot_results(self, save_path='heart_disease_evaluation.png'):
        """Create comprehensive visualizations."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return
        
        print("\n🎨 Creating visualizations...")
        
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.4f}', color='darkorange')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, y_pred_proba)
        ax3.plot(recall_vals, precision_vals, linewidth=2, color='green')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature Importance (if available)
        ax4 = fig.add_subplot(gs[1, :])
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            ax4.bar(range(len(importances)), importances[indices], color='steelblue')
            ax4.set_xticks(range(len(importances)))
            ax4.set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
            ax4.set_title('Feature Importance', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Importance Score')
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        # 5. Model Comparison
        ax5 = fig.add_subplot(gs[2, 0])
        model_names = list(self.model_performance.keys())
        f1_scores = [self.model_performance[m]['f1_score'] for m in model_names]
        colors = ['gold' if m == self.best_model_name else 'skyblue' for m in model_names]
        ax5.barh(model_names, f1_scores, color=colors)
        ax5.set_xlabel('F1 Score')
        ax5.set_title('Model Comparison (F1 Score)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Prediction Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.6, 
                label='Healthy', color='green', edgecolor='black')
        ax6.hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.6, 
                label='Heart Disease', color='red', edgecolor='black')
        ax6.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax6.set_xlabel('Predicted Probability')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Metrics Summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        metrics = self.evaluate_model()
        metrics_text = f"""
        Model Performance Summary
        {'='*30}
        
        Model: {self.best_model_name}
        
        Accuracy:    {metrics['accuracy']:.3f}
        Precision:   {metrics['precision']:.3f}
        Recall:      {metrics['recall']:.3f}
        F1-Score:    {metrics['f1_score']:.3f}
        AUC-ROC:     {metrics['auc_roc']:.3f}
        Specificity: {metrics['specificity']:.3f}
        
        Test Samples: {len(self.y_test)}
        """
        ax7.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved as '{save_path}'")
        plt.show()
    
    def predict(self, patient_data):
        """Make prediction for a single patient with detailed report."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return None
        
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Ensure correct feature order
        patient_df = patient_df[self.feature_names]
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Make prediction
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'diagnosis': 'Heart Disease' if prediction == 1 else 'Healthy',
            'probability': float(probability[1]),
            'confidence': float(max(probability)),
            'risk_level': self._get_risk_level(probability[1]),
            'risk_factors': self._identify_risk_factors(patient_data)
        }
        
        return result
    
    def predict_batch(self, patients_data):
        """Make predictions for multiple patients."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return None
        
        print(f"\n🔮 Processing {len(patients_data)} patients...")
        
        results = []
        for i, patient in enumerate(patients_data, 1):
            result = self.predict(patient)
            result['patient_id'] = i
            results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"✅ Batch prediction completed!")
        
        return results_df
    
    def generate_patient_report(self, patient_data):
        """Generate detailed patient report."""
        print("\n" + "="*70)
        print("📋 PATIENT HEALTH REPORT")
        print("="*70)
        
        result = self.predict(patient_data)
        
        print(f"\n🔬 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 Model Used: {self.best_model_name}")
        
        print("\n📊 Patient Data:")
        for feature, value in patient_data.items():
            print(f"   {feature}: {value}")
        
        print("\n🎯 Diagnosis Results:")
        print(f"   Status: {result['diagnosis']}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"   Disease Probability: {result['probability']*100:.1f}%")
        print(f"   Risk Level: {result['risk_level']}")
        
        print("\n⚠️  Identified Risk Factors:")
        if result['risk_factors']:
            for i, factor in enumerate(result['risk_factors'], 1):
                print(f"   {i}. {factor}")
        else:
            print("   ✅ No major risk factors identified")
        
        print("\n💡 Recommendations:")
        if result['prediction'] == 1:
            print("   ⚠️  URGENT: Consult a cardiologist immediately")
            print("   📋 Further tests recommended:")
            print("      - ECG/EKG")
            print("      - Echocardiogram")
            print("      - Coronary angiography")
            print("      - Stress test")
        else:
            if result['probability'] > 0.3:
                print("   ⚠️  Moderate risk detected")
                print("   📋 Preventive measures:")
                print("      - Regular health checkups")
                print("      - Monitor blood pressure and cholesterol")
                print("      - Maintain healthy lifestyle")
            else:
                print("   ✅ Low risk - Continue healthy lifestyle")
                print("   📋 General recommendations:")
                print("      - Annual health checkups")
                print("      - Regular exercise")
                print("      - Balanced diet")
        
        print("\n" + "="*70)
        print("⚠️  DISCLAIMER: This is an AI prediction tool.")
        print("Always consult healthcare professionals for medical decisions.")
        print("="*70)
        
        return result
    
    def _get_risk_level(self, probability):
        """Categorize risk level based on probability."""
        if probability < 0.25:
            return 'Low Risk'
        elif probability < 0.50:
            return 'Moderate Risk'
        elif probability < 0.75:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def _identify_risk_factors(self, patient_data):
        """Identify potential risk factors from patient data."""
        risk_factors = []
        
        # Age risk
        if patient_data['Age'] > 55:
            risk_factors.append("Advanced age (>55 years)")
        
        # Blood pressure
        if patient_data['RestingBP'] > 140:
            risk_factors.append("High blood pressure (Hypertension)")
        
        # Cholesterol
        if patient_data['Cholesterol'] > 240:
            risk_factors.append("High cholesterol (>240 mg/dl)")
        
        # Fasting blood sugar
        if patient_data['FastingBS'] == 1:
            risk_factors.append("Elevated fasting blood sugar (Diabetes indicator)")
        
        # Exercise angina
        if patient_data['ExerciseAngina'] == 1:
            risk_factors.append("Exercise-induced angina")
        
        # Low max heart rate
        if patient_data['MaxHR'] < 120:
            risk_factors.append("Low maximum heart rate")
        
        # Oldpeak
        if patient_data['Oldpeak'] > 2.0:
            risk_factors.append("Significant ST depression")
        
        # Gender
        if patient_data['Sex'] == 1:
            risk_factors.append("Male gender (higher risk)")
        
        return risk_factors
    
    def save_model(self, filepath='models/heart_disease_model.pkl'):
        """Save the trained model and scaler to disk."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model, scaler, and metadata
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'best_model_name': self.best_model_name,
                'model_performance': self.model_performance,
                'models_dict': self.models_dict
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved successfully to '{filepath}'")
            return True
        
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath='models/heart_disease_model.pkl'):
        """Load a pre-trained model from disk."""
        try:
            if not os.path.exists(filepath):
                print(f"❌ Error: Model file not found at '{filepath}'")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.best_model_name = model_data['best_model_name']
            self.model_performance = model_data['model_performance']
            self.models_dict = model_data['models_dict']
            self.is_trained = True
            
            print(f"✅ Model loaded successfully from '{filepath}'")
            print(f"🤖 Loaded Model: {self.best_model_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def interactive_prediction(self):
        """Interactive mode for making predictions."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return
        
        print("\n" + "="*70)
        print("🏥 INTERACTIVE HEART DISEASE PREDICTION")
        print("="*70)
        print("\nPlease enter patient information:\n")
        
        try:
            # Collect patient data
            age = int(input("Age (29-80): "))
            sex = int(input("Sex (0=Female, 1=Male): "))
            chest_pain = int(input("Chest Pain Type (0-3): "))
            resting_bp = int(input("Resting Blood Pressure (90-200): "))
            cholesterol = int(input("Cholesterol (100-600): "))
            fasting_bs = int(input("Fasting Blood Sugar >120 mg/dl (0=No, 1=Yes): "))
            resting_ecg = int(input("Resting ECG (0=Normal, 1=ST-T abnormality, 2=LVH): "))
            max_hr = int(input("Maximum Heart Rate (60-220): "))
            exercise_angina = int(input("Exercise Induced Angina (0=No, 1=Yes): "))
            oldpeak = float(input("ST Depression (0.0-6.2): "))
            st_slope = int(input("ST Slope (0=Up, 1=Flat, 2=Down): "))
            
            # Create patient data dictionary
            patient_data = {
                'Age': age,
                'Sex': sex,
                'ChestPainType': chest_pain,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs,
                'RestingECG': resting_ecg,
                'MaxHR': max_hr,
                'ExerciseAngina': exercise_angina,
                'Oldpeak': oldpeak,
                'ST_Slope': st_slope
            }
            
            # Generate report
            self.generate_patient_report(patient_data)
            
        except ValueError as e:
            print(f"\n❌ Invalid input: {e}")
            print("Please enter numeric values only.")
        except KeyboardInterrupt:
            print("\n\n👋 Prediction cancelled by user.")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("🏥 ADVANCED HEART DISEASE PREDICTION SYSTEM")
    print("="*70)
    print("Author: Advanced Medical AI")
    print("Date: November 2025")
    print("="*70)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load and explore data
    predictor.load_data()
    predictor.explore_data()
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Train multiple models
    predictor.train_multiple_models()
    
    # Create ensemble model
    predictor.create_ensemble_model()
    
    # Compare models
    predictor.compare_models()
    
    # Evaluate best model
    predictor.evaluate_model()
    
    # Create visualizations
    predictor.plot_results()
    
    # Save model
    predictor.save_model()
    
    # Example predictions
    print("\n" + "="*70)
    print("📊 EXAMPLE PREDICTIONS")
    print("="*70)
    
    # Example 1: High risk patient
    print("\n🔴 Example 1: High Risk Patient")
    high_risk_patient = {
        'Age': 65,
        'Sex': 1,
        'ChestPainType': 3,
        'RestingBP': 160,
        'Cholesterol': 280,
        'FastingBS': 1,
        'RestingECG': 1,
        'MaxHR': 110,
        'ExerciseAngina': 1,
        'Oldpeak': 3.5,
        'ST_Slope': 2
    }
    predictor.generate_patient_report(high_risk_patient)
    
    # Example 2: Low risk patient
    print("\n🟢 Example 2: Low Risk Patient")
    low_risk_patient = {
        'Age': 35,
        'Sex': 0,
        'ChestPainType': 0,
        'RestingBP': 115,
        'Cholesterol': 190,
        'FastingBS': 0,
        'RestingECG': 0,
        'MaxHR': 175,
        'ExerciseAngina': 0,
        'Oldpeak': 0.0,
        'ST_Slope': 1
    }
    predictor.generate_patient_report(low_risk_patient)
    
    # Batch prediction example
    print("\n" + "="*70)
    print("📊 BATCH PREDICTION EXAMPLE")
    print("="*70)
    
    batch_patients = [
        {'Age': 55, 'Sex': 1, 'ChestPainType': 2, 'RestingBP': 140, 'Cholesterol': 230,
         'FastingBS': 0, 'RestingECG': 1, 'MaxHR': 130, 'ExerciseAngina': 1,
         'Oldpeak': 2.0, 'ST_Slope': 1},
        {'Age': 42, 'Sex': 0, 'ChestPainType': 0, 'RestingBP': 120, 'Cholesterol': 200,
         'FastingBS': 0, 'RestingECG': 0, 'MaxHR': 165, 'ExerciseAngina': 0,
         'Oldpeak': 0.5, 'ST_Slope': 0},
        {'Age': 70, 'Sex': 1, 'ChestPainType': 3, 'RestingBP': 170, 'Cholesterol': 300,
         'FastingBS': 1, 'RestingECG': 2, 'MaxHR': 95, 'ExerciseAngina': 1,
         'Oldpeak': 4.0, 'ST_Slope': 2}
    ]
    
    batch_results = predictor.predict_batch(batch_patients)
    print("\n" + batch_results.to_string())
    
    # Interactive mode (commented out for automated execution)
    # print("\n" + "="*70)
    # print("🎯 INTERACTIVE MODE")
    # print("="*70)
    # while True:
    #     response = input("\nWould you like to make an interactive prediction? (yes/no): ")
    #     if response.lower() in ['yes', 'y']:
    #         predictor.interactive_prediction()
    #     else:
    #         break
    
    print("\n" + "="*70)
    print("✅ PROGRAM COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n📁 Generated Files:")
    print("  - heart_disease_evaluation.png (Visualizations)")
    print("  - models/heart_disease_model.pkl (Saved Model)")
    print("\n⚠️ MEDICAL DISCLAIMER:")
    print("This AI tool is for educational and research purposes only.")
    print("Always consult qualified healthcare professionals for medical decisions.")
    print("="*70)


if __name__ == "__main__":
    main()
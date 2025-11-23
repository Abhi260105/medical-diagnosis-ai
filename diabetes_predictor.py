"""
Medical Diagnosis AI - Diabetes Prediction Module
Author: Your Name
Date: November 2025
Description: Machine learning model for diabetes prediction using patient health data
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class DiabetesPredictor:
    """
    A comprehensive diabetes prediction system using multiple ML algorithms.
    
    Features:
    - Data preprocessing and validation
    - Multiple model training and comparison
    - Hyperparameter tuning
    - Model evaluation and visualization
    - Prediction with confidence scores
    """
    
    def __init__(self):
        """Initialize the predictor with default parameters."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.is_trained = False
        
    def load_data(self, filepath='data/diabetes.csv'):
        """
        Load diabetes dataset from CSV file.
        
        Args:
            filepath (str): Path to the diabetes dataset
            
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
        """Create a sample dataset for demonstration purposes."""
        print("🔄 Generating synthetic dataset...")
        np.random.seed(42)
        n_samples = 768
        
        data = []
        for i in range(n_samples):
            # Determine if patient has diabetes (35% positive cases)
            has_diabetes = np.random.random() < 0.35
            
            if has_diabetes:
                # Diabetic patient characteristics
                pregnancies = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                glucose = int(np.random.normal(140, 30))
                blood_pressure = int(np.random.normal(75, 12))
                skin_thickness = int(np.random.normal(32, 10))
                insulin = int(np.random.normal(150, 80))
                bmi = round(np.random.normal(33, 6), 1)
                dpf = round(np.random.uniform(0.3, 2.0), 3)
                age = int(np.random.normal(45, 15))
            else:
                # Non-diabetic patient characteristics
                pregnancies = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
                glucose = int(np.random.normal(110, 20))
                blood_pressure = int(np.random.normal(70, 10))
                skin_thickness = int(np.random.normal(28, 8))
                insulin = int(np.random.normal(100, 50))
                bmi = round(np.random.normal(28, 5), 1)
                dpf = round(np.random.uniform(0.1, 1.0), 3)
                age = int(np.random.normal(35, 12))
            
            # Ensure realistic ranges
            glucose = max(0, min(199, glucose))
            blood_pressure = max(0, min(122, blood_pressure))
            skin_thickness = max(0, min(99, skin_thickness))
            insulin = max(0, min(846, insulin))
            bmi = max(0.0, min(67.1, bmi))
            dpf = max(0.078, min(2.42, dpf))
            age = max(21, min(81, age))
            
            # Add missing values
            if np.random.random() < 0.05:
                glucose = 0
            if np.random.random() < 0.20:
                blood_pressure = 0
            if np.random.random() < 0.30:
                skin_thickness = 0
            if np.random.random() < 0.48:
                insulin = 0
            
            data.append([pregnancies, glucose, blood_pressure, skin_thickness, 
                        insulin, bmi, dpf, age, 1 if has_diabetes else 0])
        
        df = pd.DataFrame(data, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ])
        
        print(f"✅ Generated {len(df)} samples")
        return df
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*50)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print(f"\n1️⃣ Dataset Shape: {self.df.shape}")
        print(f"   Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n2️⃣ Statistical Summary:")
        print(self.df.describe().round(2))
        
        print("\n3️⃣ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\n4️⃣ Class Distribution:")
        counts = self.df['Outcome'].value_counts()
        print(f"   Diabetic Cases (1): {counts.get(1, 0)} ({counts.get(1, 0)/len(self.df)*100:.2f}%)")
        print(f"   Healthy Cases (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(self.df)*100:.2f}%)")
        
    def preprocess_data(self):
        """
        Preprocess the data for machine learning.
        """
        print("\n" + "="*50)
        print("🔧 DATA PREPROCESSING")
        print("="*50)
        
        # Replace zeros with NaN for specific features
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for feature in zero_features:
            self.df[feature] = self.df[feature].replace(0, np.nan)
        
        # Fill missing values with median
        self.df.fillna(self.df.median(), inplace=True)
        print("✅ Missing values handled")
        
        # Separate features and target
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("✅ Features scaled using StandardScaler")
        
    def train_model(self, algorithm='random_forest', tune_hyperparameters=False):
        """
        Train a machine learning model.
        """
        print("\n" + "="*50)
        print(f"🚀 TRAINING MODEL: {algorithm.upper()}")
        print("="*50)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        self.model = models[algorithm]
        self.model.fit(self.X_train_scaled, self.y_train)
        print("✅ Model trained successfully")
        
        # Cross-validation
        print("🔄 Performing cross-validation...")
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=5, scoring='f1')
        print(f"📊 Cross-validation F1 scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"📊 Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        
    def evaluate_model(self):
        """Evaluate the trained model and display metrics."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return None
        
        print("\n" + "="*50)
        print("📈 MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall:    {recall:.4f}")
        print(f"✅ F1-Score:  {f1:.4f}")
        print(f"✅ AUC-ROC:   {auc_roc:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Healthy', 'Diabetic']))
        
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm
        }
    
    def plot_results(self, save_path='model_evaluation.png'):
        """Visualize model performance."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return
        
        print("\n🎨 Creating visualizations...")
        
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.4f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            axes[1, 0].bar(range(len(importances)), importances[indices])
            axes[1, 0].set_xticks(range(len(importances)))
            axes[1, 0].set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
            axes[1, 0].set_title('Feature Importance', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Importance')
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available for this model', 
                          ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        # Prediction Distribution
        axes[1, 1].hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.5, label='Healthy', color='green')
        axes[1, 1].hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.5, label='Diabetic', color='red')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved as '{save_path}'")
        plt.show()
    
    def predict(self, patient_data):
        """
        Make a prediction for a single patient.
        """
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
            'diagnosis': 'Diabetic' if prediction == 1 else 'Healthy',
            'probability': float(probability[1]),
            'confidence': float(max(probability)),
            'risk_level': self._get_risk_level(probability[1])
        }
        
        return result
    
    def _get_risk_level(self, probability):
        """Categorize risk level based on probability."""
        if probability < 0.3:
            return 'Low Risk'
        elif probability < 0.6:
            return 'Moderate Risk'
        elif probability < 0.8:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def save_model(self, filepath='models/diabetes_model.pkl'):
        """Save the trained model and scaler."""
        if not self.is_trained:
            print("❌ Error: Model not trained yet!")
            return
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved successfully to {filepath}")
    
    def load_model(self, filepath='models/diabetes_model.pkl'):
        """Load a pre-trained model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
            print(f"✅ Model loaded successfully from {filepath}")
            return True
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {filepath}")
            return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("🏥 MEDICAL DIAGNOSIS AI - DIABETES PREDICTION SYSTEM")
    print("="*70)
    
    try:
        # Initialize predictor
        predictor = DiabetesPredictor()
        
        # Load and explore data
        predictor.load_data()
        predictor.explore_data()
        
        # Preprocess data
        predictor.preprocess_data()
        
        # Train model
        predictor.train_model(algorithm='random_forest')
        
        # Evaluate model
        metrics = predictor.evaluate_model()
        
        # Visualize results
        try:
            predictor.plot_results()
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")
            print("   (This is normal if running in non-GUI environment)")
        
        # Save model
        predictor.save_model()
        
        # Example prediction
        print("\n" + "="*70)
        print("🔮 EXAMPLE PREDICTION")
        print("="*70)
        
        sample_patient = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 79,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        
        print("\n📋 Patient Data:")
        for key, value in sample_patient.items():
            print(f"  {key}: {value}")
        
        result = predictor.predict(sample_patient)
        
        print("\n🎯 Prediction Results:")
        print(f"  Diagnosis: {result['diagnosis']}")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        print("\n" + "="*70)
        print("✅ ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
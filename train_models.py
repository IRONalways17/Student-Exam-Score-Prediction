"""
Student Exam Score Prediction - Model Training Script
This script trains all models and saves them for deployment.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='student_performance.csv'):
    """Load and preprocess the student performance data."""
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    df_processed = df.copy()
    
    # Remove Student_ID
    df_processed = df_processed.drop('Student_ID', axis=1)
    
    # Encode categorical variables
    binary_cols = ['Participation_in_Discussions', 'Use_of_Educational_Tech']
    for col in binary_cols:
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
    
    # Label encoding for ordinal variables
    ordinal_mappings = {
        'Final_Grade': {'A': 4, 'B': 3, 'C': 2, 'D': 1},
        'Self_Reported_Stress_Level': {'Low': 1, 'Medium': 2, 'High': 3}
    }
    
    for col, mapping in ordinal_mappings.items():
        df_processed[col] = df_processed[col].map(mapping)
    
    # One-hot encoding for nominal categorical variables
    nominal_cols = ['Gender', 'Preferred_Learning_Style']
    df_processed = pd.get_dummies(df_processed, columns=nominal_cols, prefix=nominal_cols)
    
    # Feature Engineering
    df_processed['Study_Efficiency'] = df_processed['Assignment_Completion_Rate (%)'] / (df_processed['Study_Hours_per_Week'] + 1)
    df_processed['Work_Life_Balance'] = df_processed['Sleep_Hours_per_Night'] / (df_processed['Time_Spent_on_Social_Media (hours/week)'] + 1)
    df_processed['Academic_Engagement'] = (
        df_processed['Study_Hours_per_Week'] * 0.4 + 
        df_processed['Online_Courses_Completed'] * 0.3 + 
        df_processed['Participation_in_Discussions'] * 20 + 
        df_processed['Assignment_Completion_Rate (%)'] * 0.3
    )
    
    # Age groups
    df_processed['Age_Group'] = pd.cut(df_processed['Age'], bins=[17, 20, 23, 26, 30], 
                                      labels=['18-20', '21-23', '24-26', '27-30'])
    df_processed = pd.get_dummies(df_processed, columns=['Age_Group'], prefix='Age_Group')
    
    print(f"Data preprocessing completed. Shape: {df_processed.shape}")
    return df_processed

def train_models(df_processed):
    """Train all machine learning models."""
    print("Training machine learning models...")
    
    # Prepare features and target
    X = df_processed.drop('Exam_Score (%)', axis=1)
    y = df_processed['Exam_Score (%)']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=pd.qcut(y, q=5, labels=False)
    )
    
    # Scale features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }
    
    # Train models and collect results
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        trained_models[name] = model
        print(f"      R² Score: {r2:.3f}, RMSE: {rmse:.3f}")
    
    return trained_models, results, scaler, X.columns.tolist(), (X_test, y_test)

def save_models(models, scaler, feature_columns, results):
    """Save trained models and metadata."""
    print("Saving models...")
    
    # Save models
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'model_results': results,
        'best_model': max(results.items(), key=lambda x: x[1]['R²'])[0]
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Models saved successfully!")

def predict_student_score(student_data, models, scaler, feature_columns):
    """
    Predict exam score for a new student.
    
    Args:
        student_data (dict): Dictionary with student features
        models (dict): Trained models
        scaler: Fitted StandardScaler
        feature_columns (list): List of feature column names
    
    Returns:
        dict: Predictions from all models
    """
    # Convert to DataFrame and ensure all columns are present
    df_student = pd.DataFrame([student_data])
    
    # Add missing columns with default values
    for col in feature_columns:
        if col not in df_student.columns:
            df_student[col] = 0
    
    # Reorder columns
    df_student = df_student[feature_columns]
    
    # Make predictions
    predictions = {}
    for name, model in models.items():
        if name == 'Linear Regression':
            student_scaled = scaler.transform(df_student)
            pred = model.predict(student_scaled)[0]
        else:
            pred = model.predict(df_student)[0]
        
        predictions[name] = pred
    
    predictions['Average'] = np.mean(list(predictions.values()))
    return predictions

def main():
    """Main execution function."""
    print("Student Exam Score Prediction - Model Training")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        df_processed = load_and_preprocess_data()
        
        # Train models
        models, results, scaler, feature_columns, test_data = train_models(df_processed)
        
        # Display results
        print(f"\nModel Performance Summary:")
        print("-" * 50)
        for name, metrics in results.items():
            print(f"{name:<20} | R²: {metrics['R²']:.3f} | RMSE: {metrics['RMSE']:.3f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['R²'])
        print(f"\nBest Model: {best_model[0]} (R² = {best_model[1]['R²']:.3f})")
        
        # Save models
        save_models(models, scaler, feature_columns, results)
        
        # Example prediction
        print(f"\nExample Prediction:")
        sample_student = {
            'Age': 20,
            'Study_Hours_per_Week': 35,
            'Online_Courses_Completed': 12,
            'Assignment_Completion_Rate (%)': 85,
            'Attendance_Rate (%)': 90,
            'Sleep_Hours_per_Night': 8,
            'Time_Spent_on_Social_Media (hours/week)': 15,
            'Final_Grade': 3,  # B grade
            'Self_Reported_Stress_Level': 2,  # Medium
            'Participation_in_Discussions': 1,  # Yes
            'Use_of_Educational_Tech': 1  # Yes
        }
        
        # Load saved models for prediction
        with open('trained_models.pkl', 'rb') as f:
            loaded_models = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
        
        predictions = predict_student_score(sample_student, loaded_models, loaded_scaler, feature_columns)
        
        print("Sample student predictions:")
        for model, score in predictions.items():
            print(f"   {model}: {score:.1f}%")
        
        print(f"\nTraining completed successfully!")
        print(f"Files created: trained_models.pkl, scaler.pkl, model_metadata.pkl")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()

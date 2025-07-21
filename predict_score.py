"""
Student Exam Score Prediction Tool
Simple script to predict exam scores for new students using trained models.
"""

import pandas as pd
import numpy as np
import pickle
import json

def load_models():
    """Load the trained models and necessary components."""
    try:
        with open('trained_models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return models, scaler, metadata
    except FileNotFoundError:
        print("Model files not found. Please run train_models.py first.")
        return None, None, None

def create_student_profile():
    """Interactive function to create a student profile."""
    print("Enter student information:")
    print("-" * 40)
    
    student = {}
    
    # Basic demographics
    student['Age'] = int(input("Age (18-30): "))
    
    # Academic metrics
    student['Study_Hours_per_Week'] = float(input("Study hours per week (0-50): "))
    student['Online_Courses_Completed'] = int(input("Online courses completed (0-25): "))
    student['Assignment_Completion_Rate (%)'] = float(input("Assignment completion rate % (0-100): "))
    student['Attendance_Rate (%)'] = float(input("Attendance rate % (0-100): "))
    
    # Lifestyle factors
    student['Sleep_Hours_per_Night'] = float(input("Sleep hours per night (4-12): "))
    student['Time_Spent_on_Social_Media (hours/week)'] = float(input("Social media hours per week (0-40): "))
    
    # Categorical inputs
    print("\nFinal Grade (1=D, 2=C, 3=B, 4=A):")
    student['Final_Grade'] = int(input("Final Grade (1-4): "))
    
    print("\nStress Level (1=Low, 2=Medium, 3=High):")
    student['Self_Reported_Stress_Level'] = int(input("Stress level (1-3): "))
    
    print("\nParticipation in Discussions (0=No, 1=Yes):")
    student['Participation_in_Discussions'] = int(input("Participation (0/1): "))
    
    print("\nUse of Educational Technology (0=No, 1=Yes):")
    student['Use_of_Educational_Tech'] = int(input("Educational tech usage (0/1): "))
    
    return student

def preprocess_student_data(student_data, feature_columns):
    """Preprocess student data to match training format."""
    # Create base DataFrame
    df_student = pd.DataFrame([student_data])
    
    # Add engineered features
    df_student['Study_Efficiency'] = df_student['Assignment_Completion_Rate (%)'] / (df_student['Study_Hours_per_Week'] + 1)
    df_student['Work_Life_Balance'] = df_student['Sleep_Hours_per_Night'] / (df_student['Time_Spent_on_Social_Media (hours/week)'] + 1)
    df_student['Academic_Engagement'] = (
        df_student['Study_Hours_per_Week'] * 0.4 + 
        df_student['Online_Courses_Completed'] * 0.3 + 
        df_student['Participation_in_Discussions'] * 20 + 
        df_student['Assignment_Completion_Rate (%)'] * 0.3
    )
    
    # Add age group (using most common: 18-20)
    for col in ['Age_Group_18-20', 'Age_Group_21-23', 'Age_Group_24-26', 'Age_Group_27-30']:
        df_student[col] = 0
    
    if student_data['Age'] <= 20:
        df_student['Age_Group_18-20'] = 1
    elif student_data['Age'] <= 23:
        df_student['Age_Group_21-23'] = 1
    elif student_data['Age'] <= 26:
        df_student['Age_Group_24-26'] = 1
    else:
        df_student['Age_Group_27-30'] = 1
    
    # Add dummy categorical variables (using defaults)
    categorical_defaults = {
        'Gender_Female': 1, 'Gender_Male': 0, 'Gender_Other': 0,
        'Preferred_Learning_Style_Auditory': 0, 'Preferred_Learning_Style_Kinesthetic': 0,
        'Preferred_Learning_Style_Reading/Writing': 0, 'Preferred_Learning_Style_Visual': 1
    }
    
    for col, value in categorical_defaults.items():
        df_student[col] = value
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in df_student.columns:
            df_student[col] = 0
    
    # Reorder columns to match training data
    df_student = df_student[feature_columns]
    
    return df_student

def predict_exam_score(student_data, models, scaler, feature_columns):
    """Predict exam score for a student."""
    # Preprocess data
    df_student = preprocess_student_data(student_data, feature_columns)
    
    # Make predictions
    predictions = {}
    for name, model in models.items():
        if name == 'Linear Regression':
            student_scaled = scaler.transform(df_student)
            pred = model.predict(student_scaled)[0]
        else:
            pred = model.predict(df_student)[0]
        
        predictions[name] = pred
    
    predictions['Average'] = np.mean([predictions['Linear Regression'], 
                                    predictions['Random Forest'], 
                                    predictions['Gradient Boosting']])
    
    return predictions

def interpret_score(score):
    """Provide interpretation of the predicted score."""
    if score >= 90:
        return "Excellent - Outstanding performance expected"
    elif score >= 80:
        return "Very Good - Strong performance expected"
    elif score >= 70:
        return "Good - Above average performance expected"
    elif score >= 60:
        return "Satisfactory - Average performance expected"
    elif score >= 50:
        return "Below Average - May need additional support"
    else:
        return "At Risk - Immediate intervention recommended"

def provide_recommendations(student_data, predictions):
    """Provide personalized recommendations based on student data."""
    print(f"\nPERSONALIZED RECOMMENDATIONS:")
    print("-" * 50)
    
    avg_score = predictions['Average']
    
    # Study hours recommendation
    if student_data['Study_Hours_per_Week'] < 25:
        print("Increase study hours to 25-35 hours per week for better results")
    
    # Assignment completion
    if student_data['Assignment_Completion_Rate (%)'] < 80:
        print("Focus on completing assignments - aim for 85%+ completion rate")
    
    # Attendance
    if student_data['Attendance_Rate (%)'] < 85:
        print("Improve class attendance - target 90%+ attendance")
    
    # Sleep
    if student_data['Sleep_Hours_per_Night'] < 7:
        print("Get more sleep - aim for 7-8 hours per night")
    elif student_data['Sleep_Hours_per_Night'] > 9:
        print("Consider optimizing sleep schedule - 7-8 hours is optimal")
    
    # Social media
    if student_data['Time_Spent_on_Social_Media (hours/week)'] > 20:
        print("Reduce social media usage - limit to 15 hours per week or less")
    
    # Participation
    if student_data['Participation_in_Discussions'] == 0:
        print("Participate actively in class discussions")
    
    # Educational technology
    if student_data['Use_of_Educational_Tech'] == 0:
        print("Utilize educational technology tools for better learning")
    
    # Stress management
    if student_data['Self_Reported_Stress_Level'] >= 3:
        print("Consider stress management techniques and seek support if needed")

def main():
    """Main execution function."""
    print("STUDENT EXAM SCORE PREDICTOR")
    print("=" * 50)
    
    # Load models
    models, scaler, metadata = load_models()
    if models is None:
        return
    
    feature_columns = metadata['feature_columns']
    best_model = metadata['best_model']
    
    print(f"Models loaded successfully!")
    print(f"Best performing model: {best_model}")
    print(f"Model accuracy: R² = {metadata['model_results'][best_model]['R²']:.3f}")
    
    while True:
        print(f"\n" + "=" * 50)
        print("Choose an option:")
        print("1. Enter student data manually")
        print("2. Use sample student data")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == '1':
            student_data = create_student_profile()
        elif choice == '2':
            # Sample data for testing
            student_data = {
                'Age': 20,
                'Study_Hours_per_Week': 30,
                'Online_Courses_Completed': 10,
                'Assignment_Completion_Rate (%)': 80,
                'Attendance_Rate (%)': 85,
                'Sleep_Hours_per_Night': 7,
                'Time_Spent_on_Social_Media (hours/week)': 15,
                'Final_Grade': 3,  # B
                'Self_Reported_Stress_Level': 2,  # Medium
                'Participation_in_Discussions': 1,  # Yes
                'Use_of_Educational_Tech': 1  # Yes
            }
            print("Using sample student data...")
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        
        # Make prediction
        try:
            predictions = predict_exam_score(student_data, models, scaler, feature_columns)
            
            print(f"\nEXAM SCORE PREDICTIONS:")
            print("-" * 40)
            for model, score in predictions.items():
                print(f"{model:<20}: {score:.1f}%")
            
            avg_score = predictions['Average']
            interpretation = interpret_score(avg_score)
            print(f"\n{interpretation}")
            
            # Provide recommendations
            provide_recommendations(student_data, predictions)
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()

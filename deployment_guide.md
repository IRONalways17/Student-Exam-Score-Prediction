# Student Exam Score Prediction - Streamlit Deployment Guide

## Streamlit App Deployment Instructions

### Prerequisites
- Python 3.8 or higher
- All project files in the same directory
- Student performance dataset (CSV file)

### Step 1: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 2: Train the Models
Before running the Streamlit app, you need to train and save the models:
```bash
python train_models.py
```

This will create three files:
- `trained_models.pkl` - Contains all trained ML models
- `scaler.pkl` - Feature scaling transformer
- `model_metadata.pkl` - Model performance metrics and metadata

### Step 3: Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Step 4: Using the Application

#### Input Features:
- **Demographics**: Age, Gender
- **Academic**: Study hours, assignment completion, attendance
- **Engagement**: Class participation, technology usage
- **Lifestyle**: Sleep hours, social media time, stress levels
- **History**: Previous grades, learning style

#### Application Features:
- Real-time exam score predictions
- Multiple model comparisons
- Performance visualizations
- Improvement recommendations
- Interactive charts and graphs


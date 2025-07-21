#!/bin/bash

echo "Student Exam Score Prediction - Streamlit App"
echo "================================================"
echo

echo "Step 1: Installing required packages..."
pip install -r requirements.txt
echo

echo "Step 2: Training machine learning models..."
python train_models.py
echo

echo "Step 3: Starting Streamlit application..."
echo "The app will open automatically in your browser at http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

streamlit run streamlit_app.py

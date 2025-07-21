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

## Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and main file (`streamlit_app.py`)
5. Deploy automatically

### Option 2: Heroku Deployment
1. Create these additional files:

**Procfile:**
```
web: sh setup.sh && streamlit run streamlit_app.py
```

**setup.sh:**
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

2. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

### Option 3: Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Train models during build
RUN python train_models.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t student-predictor .
docker run -p 8501:8501 student-predictor
```

## 🔧 Configuration Options

### Custom Port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Hide Streamlit Menu
Add to `streamlit_app.py`:
```python
hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)
```

### Custom Theme
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Mobile Responsiveness
The app is automatically mobile-responsive through Streamlit's built-in responsive design.

## 🔒 Security Considerations
- No sensitive data is stored
- All processing happens client-side or on your server
- Consider adding authentication for production use
- Use HTTPS in production deployments

## 🐛 Troubleshooting

### Common Issues:

**"Model files not found" error:**
```bash
# Solution: Train models first
python train_models.py
```

**Import errors:**
```bash
# Solution: Install all requirements
pip install -r requirements.txt
```

**Port already in use:**
```bash
# Solution: Use different port
streamlit run streamlit_app.py --server.port 8502
```

**Memory issues:**
- Reduce model complexity in `train_models.py`
- Use smaller dataset for testing
- Consider using cloud deployment with more memory

## Performance Monitoring
Monitor your app's performance:
- Check Streamlit metrics in the browser
- Monitor server resources (CPU, memory)
- Use cloud provider monitoring tools

## 🔄 Updates and Maintenance
To update the model:
1. Update dataset or features
2. Run `python train_models.py`
3. Restart the Streamlit app
4. Models will automatically use new trained versions

## Scaling Considerations
For high-traffic deployment:
- Use caching (`@st.cache_data`, `@st.cache_resource`)
- Consider model serving with APIs (FastAPI + Streamlit)
- Use load balancers for multiple instances
- Implement model versioning

## Success Metrics
Monitor these metrics for your deployment:
- Prediction accuracy vs actual scores
- User engagement time
- Feature usage patterns
- Error rates and performance

Your Streamlit app is now ready for deployment!

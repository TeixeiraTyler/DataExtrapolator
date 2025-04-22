# ML Prediction API

A FastAPI service that processes CSV files and trains machine learning models for prediction.

## Features

- CSV file processing with automatic handling of missing values
- Feature selection and scaling
- Outlier detection and removal
- Multiple model training (Linear Regression, Random Forest, Gradient Boosting)
- Hyperparameter tuning with GridSearchCV
- Model evaluation with multiple metrics
- Feature importance visualization

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python run.py
   ```

3. The API will be available at http://localhost:8000

## API Endpoints

### POST /api/process-csv
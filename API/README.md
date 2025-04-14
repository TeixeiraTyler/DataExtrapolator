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

Process a CSV file and train machine learning models.

- **Request**: Form data with a file field named "file" containing a CSV file
- **Response**: JSON with model results including:
  - Dropped columns (columns with >25% missing values)
  - Selected features
  - Model scores for each model type
  - Best parameters for the Random Forest model
  - Final model metrics (MAE, MSE, RMSE, R2)
  - Feature importance plot

## API Documentation

Once the application is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
.
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── container.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── ml_service.py
│   ├── __init__.py
│   └── main.py
├── tests/
├── requirements.txt
├── README.md
└── run.py
```

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- Pandas: Data manipulation
- NumPy: Numerical operations
- Scikit-learn: Machine learning
- Matplotlib: Plotting
- SciPy: Statistical operations
- Dependency-injector: Dependency injection 
# Data Extrapolator

A full-stack data analysis and machine learning application for processing CSV data, training machine learning models, and visualizing predictions.

## Features

- Upload and process CSV files with automatic handling of missing values
- Data analysis and visualization
- Feature selection and engineering
- Multiple machine learning model training:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Hyperparameter tuning
- Model evaluation with multiple metrics
- Feature importance visualization

## Architecture

This project consists of two main components:

- **API**: A FastAPI backend service for data processing and ML model training
- **UI**: A React.js frontend for user interaction and visualization

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Installation and Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd DataExtrapolator
   ```

2. Start the application with Docker Compose:
   ```
   docker-compose up --build
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Usage

1. Upload a CSV file through the web interface
2. Configure analysis and prediction parameters
3. View the results, including:
   - Data visualizations
   - Model performance metrics
   - Feature importance
   - Predictions

## Contact

tyler.teixeira@gmail.com
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional

from src.utils.graphs import GraphUtils

from src.config import Settings


class MLService:
    """Service for machine learning operations."""
    
    def __init__(self, config: Settings):
        self.config = config
    
    def remove_outliers(self, X: np.ndarray, y: np.ndarray, z_threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers based on z-score
        
        Args:
            X: Feature matrix
            y: Target variable
            z_threshold: Z-score threshold for outlier detection
        
        Returns:
            Cleaned X and y arrays with outliers removed
        """
        if z_threshold is None:
            z_threshold = self.config.Z_THRESHOLD
            
        z_scores = np.abs(stats.zscore(y))
        mask = z_scores < z_threshold
        return X[mask], y[mask]
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using multiple metrics
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
        
        Returns:
            Dictionary with evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2)
        }
    
    def process_csv(self, csv_content: bytes, dependent_variable: str, num_features: int = 10) -> Dict[str, Any]:
        """Process CSV data and train models
        
        Args:
            csv_content: CSV file content as bytes
            dependent_variable: Name of the target variable/column for prediction
            
        Returns:
            Dictionary with model results and metrics
        """
        # Load the dataset
        df = pd.read_csv(io.BytesIO(csv_content))
        
        # Calculate percentage of missing values for each column
        missing_percentages = df.isnull().sum() / len(df) * 100
        
        # Drop columns with more than 25% missing values
        columns_to_drop = missing_percentages[missing_percentages > 25].index
        df.drop(columns=columns_to_drop, inplace=True)
        
        # Handle missing values
        # Fill numeric columns with mean values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        # Fill categorical columns with mode (most frequent value)
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        # Convert categorical variables to dummy/indicator variables
        df = pd.get_dummies(df)
        
        # Convert boolean columns to uint8 to avoid numpy warnings
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype('uint8')
        
        # Check if dependent variable exists in dataframe
        if dependent_variable not in df.columns:
            raise ValueError(f"Dependent variable '{dependent_variable}' not found in the CSV file")
        
        # Separate features and target variable
        x = df[df.columns[(df.columns != dependent_variable) & (df.columns != 'Id')]]
        y = df[dependent_variable]
        
        # Feature selection - Select most important features
        best_features = SelectKBest(score_func=f_regression, k=num_features)
        modified_x = best_features.fit_transform(x, y)
        
        # Get names of selected features
        selected_features_mask = best_features.get_support()
        selected_feature_names = x.columns[selected_features_mask].tolist()
        
        # Create DataFrame with selected features
        modified_x = pd.DataFrame(modified_x, columns=selected_feature_names)
        
        # Scale the features
        scaler = StandardScaler()
        modified_x = scaler.fit_transform(modified_x)
        modified_x = np.array(modified_x)
        
        # Remove outliers from the dataset
        modified_x_clean, y_clean = self.remove_outliers(modified_x, y)
        
        # Define different models to try
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=self.config.RANDOM_STATE),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.config.RANDOM_STATE)
        }
        
        # Evaluate each model using cross-validation
        model_scores = {}
        for name, model in models.items():
            scores = cross_val_score(model, modified_x_clean, y_clean, cv=5, scoring='r2')
            model_scores[name] = {
                "average_r2": float(scores.mean()),
                "std_r2": float(scores.std() * 2)
            }
        
        # Perform hyperparameter tuning on Random Forest (best performing model)
        best_model = RandomForestRegressor(random_state=self.config.RANDOM_STATE)
        param_grid = {
            'n_estimators': self.config.N_ESTIMATORS,
            'max_depth': self.config.MAX_DEPTH,
            'min_samples_split': self.config.MIN_SAMPLES_SPLIT
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(modified_x_clean, y_clean)
        
        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            modified_x_clean, y_clean, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE
        )
        
        # Train final model with best parameters
        final_model = grid_search.best_estimator_
        final_model.fit(x_train, y_train)
        y_pred = final_model.predict(x_test)
        
        # Evaluate final model performance
        final_metrics = self.evaluate_model(y_test, y_pred)
        
        correlation_plot = GraphUtils.plot_feature_correlations(df, dependent_variable, selected_feature_names)
        distibution_plot = GraphUtils.plot_feature_distributions(df, selected_feature_names)
        matrix_plot = GraphUtils.plot_scatter_matrix(df, selected_feature_names)
        importance_plot = GraphUtils.plot_feature_importance(final_model, selected_feature_names)
        residual_plot = GraphUtils.plot_residuals(y_test, y_pred)
        
        return {
            "dropped_columns": list(columns_to_drop),
            "selected_features": selected_feature_names,
            "model_scores": model_scores,
            "best_parameters": grid_search.best_params_,
            "best_cross_validation_score": float(grid_search.best_score_),
            "final_metrics": final_metrics,
            "correlation_plot": correlation_plot,
            "distibution_plot": distibution_plot,
            "matrix_plot": matrix_plot,
            "importance_plot": importance_plot,
            "residual_plot": residual_plot
        }

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Any
from scipy import stats

class GraphUtils:
    @staticmethod
    def plot_feature_correlations(df: pd.DataFrame, target_variable: str, selected_feature_names: List[str], figsize: Tuple[int, int] = (12, 8)) -> bytes:
        """Create correlation plots between features and target variable
        
        Args:
            df: Input DataFrame containing features and target
            target_variable: Name of the target variable column
            best_features: SelectKBest object containing the selected features
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        # Calculate correlations with target variable
        correlations = df.corr()[target_variable]
        
        # Filter correlations to only include selected features
        correlations = correlations[correlations.index.isin(selected_feature_names)]
        
        # Use absolute correlation values
        correlations = correlations.abs()
        
        # Sort by absolute correlation values (descending)
        correlations = correlations.sort_values(ascending=False)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Create bar plot
        bars = plt.bar(range(len(correlations)), correlations.values)
        
        # Customize the plot
        plt.title(f'Absolute Feature Correlations with {target_variable}', pad=20)
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation Coefficient')
        plt.xticks(range(len(correlations)), correlations.index, rotation=45, ha='right')
        
        # Add correlation values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reset buffer position
        buf.seek(0)
        return buf.getvalue() 

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (12, 8)) -> bytes:
        """Create a residual plot showing the difference between predicted and actual values
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=figsize)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def plot_feature_distributions(df: pd.DataFrame, selected_features: List[str], figsize: Tuple[int, int] = (15, 10)) -> bytes:
        """Create distribution plots for selected features
        
        Args:
            df: Input DataFrame
            selected_features: List of feature names to plot
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        n_features = len(selected_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=figsize)
        for i, feature in enumerate(selected_features, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.hist(df[feature], bins=30, alpha=0.7)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def plot_box_plots(df: pd.DataFrame, selected_features: List[str], figsize: Tuple[int, int] = (15, 10)) -> bytes:
        """Create box plots for selected features
        
        Args:
            df: Input DataFrame
            selected_features: List of feature names to plot
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        plt.figure(figsize=figsize)
        df[selected_features].boxplot()
        plt.title('Box Plots of Selected Features')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def plot_scatter_matrix(df: pd.DataFrame, selected_features: List[str], figsize: Tuple[int, int] = (15, 15)) -> bytes:
        """Create a scatter plot matrix for selected features
        
        Args:
            df: Input DataFrame
            selected_features: List of feature names to plot
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        plt.figure(figsize=figsize)
        pd.plotting.scatter_matrix(df[selected_features], alpha=0.5, figsize=figsize)
        plt.suptitle('Scatter Plot Matrix of Selected Features', y=1.02)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def plot_learning_curves(model: Any, X: np.ndarray, y: np.ndarray, 
                           train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                           figsize: Tuple[int, int] = (10, 6)) -> bytes:
        """Create learning curves for a given model
        
        Args:
            model: Scikit-learn model
            X: Feature matrix
            y: Target variable
            train_sizes: Array of training set sizes
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5, scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.title('Learning Curves')
        plt.xlabel('Training Examples')
        plt.ylabel('R² Score')
        plt.legend(loc='best')
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def plot_feature_importance(model: Any, feature_names: List[str], figsize: Tuple[int, int] = (12, 8)) -> bytes:
        """Create a feature importance plot for tree-based models
        
        Args:
            model: Tree-based model (RandomForest, GradientBoosting, etc.)
            feature_names: List of feature names
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
            
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=figsize)
        plt.title('Feature Importance')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (10, 6)) -> bytes:
        """Create a scatter plot of predicted vs actual values
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def plot_qq(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (10, 6)) -> bytes:
        """Create a QQ plot of residuals
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            figsize: Tuple of figure dimensions (width, height)
            
        Returns:
            Bytes object containing the plot image
        """
        residuals = y_true - y_pred
        plt.figure(figsize=figsize)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()
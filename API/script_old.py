import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LinearRegression  # For linear regression model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # For model evaluation metrics
from sklearn.feature_selection import SelectKBest  # For feature selection
from sklearn.feature_selection import f_regression  # For feature selection scoring
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # For ensemble models
from sklearn.model_selection import GridSearchCV, cross_val_score  # For model tuning and validation
from scipy import stats  # For statistical operations

def remove_outliers(X, y, z_threshold=3):
    """Remove outliers based on z-score
    
    Args:
        X: Feature matrix
        y: Target variable
        z_threshold: Z-score threshold for outlier detection (default=3)
    
    Returns:
        Cleaned X and y arrays with outliers removed
    """
    z_scores = np.abs(stats.zscore(y))
    mask = z_scores < z_threshold
    return X[mask], y[mask]

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
    
    Prints:
        MAE: Mean Absolute Error
        MSE: Mean Squared Error
        RMSE: Root Mean Squared Error
        R2: R-squared score
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.3f}")

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the model
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_)
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Load the dataset
df = pd.read_csv('test_data/train.csv')

# Calculate percentage of missing values for each column
missing_percentages = df.isnull().sum() / len(df) * 100

# Drop columns with more than 25% missing values
columns_to_drop = missing_percentages[missing_percentages > 25].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns with >25% missing values: {list(columns_to_drop)}")

# Handle missing values
# Fill numeric columns with mean values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
# Fill categorical columns with mode (most frequent value)
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df)

# Separate features and target variable
x = df[df.columns[(df.columns != 'SalePrice') & (df.columns != 'Id')]]
y = df['SalePrice']

# Feature selection - Select top 15 most important features
best_features = SelectKBest(score_func=f_regression, k=15)
modified_x = best_features.fit_transform(x, y)

# Get names of selected features
selected_features_mask = best_features.get_support()
selected_feature_names = x.columns[selected_features_mask].tolist()
print("Selected features:", selected_feature_names)

# Create DataFrame with selected features
modified_x = pd.DataFrame(modified_x, columns=selected_feature_names)

# Scale the features
scaler = StandardScaler()
modified_x = scaler.fit_transform(modified_x)
modified_x = np.array(modified_x)

# Remove outliers from the dataset
modified_x_clean, y_clean = remove_outliers(modified_x, y)

# Define different models to try
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=0),
    'Gradient Boosting': GradientBoostingRegressor(random_state=0)
}

# Evaluate each model using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, modified_x_clean, y_clean, cv=5, scoring='r2')
    print(f"{name} - Average R2: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Perform hyperparameter tuning on Random Forest (best performing model)
best_model = RandomForestRegressor(random_state=0)
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [10, 20, 30, None],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10]   # Minimum samples required to split
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(modified_x_clean, y_clean)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(modified_x_clean, y_clean, test_size=0.2, random_state=0)

# Train final model with best parameters
final_model = grid_search.best_estimator_
final_model.fit(x_train, y_train)
y_pred = final_model.predict(x_test)

# Evaluate final model performance
print("\nFinal Model Evaluation:")
evaluate_model(y_test, y_pred)

# Plot feature importance for the final model
plot_feature_importance(final_model, selected_feature_names)

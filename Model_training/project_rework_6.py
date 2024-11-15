from project_rework_1 import load_dataframe, clean_missing_data, drop_column, print_dataframe_summary, save_dataframe
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import json
import os
import pickle
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from typing import Dict, Any
import joblib
import random


def analyze_correlation(df: pd.DataFrame, target_column: str = 'Price', correlation_threshold: float = 0.25, plot_heatmap: bool = True, plot_scatter: bool = True, columns_to_exclude: list = None) -> pd.DataFrame:
    """
    Analyzes the correlation between features and the target variable, and optionally displays a correlation heatmap and scatter plots.

    Args:
        df (pd.DataFrame): The pre-loaded DataFrame.
        target_column (str): The target column to calculate correlation against. Default is 'Price'.
        correlation_threshold (float): Minimum correlation threshold for feature selection. Default is 0.25.
        plot_heatmap (bool): Whether to display the correlation heatmap. Default is True.
        plot_scatter (bool): Whether to display scatter plots for each feature against the target variable. Default is True.
        columns_to_exclude (list): List of column names to exclude from analysis. If None, no columns are excluded.

    Returns:
        pd.DataFrame: DataFrame containing only features with correlation above the threshold with the target variable.
    """
    
    # Check if the target column exists
    if target_column not in df.columns:
        raise ValueError(f"The target column '{target_column}' is not in the DataFrame.")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    if not missing_columns.empty:
        print("Missing values found in the following columns:")
        print(missing_columns)
    else:
        print("No missing values found in the dataset.")
    
    # Exclude specific columns from processing, only if columns_to_exclude is provided
    if columns_to_exclude is not None:
        df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns], errors='ignore')

    # Select only numeric columns (int, float)
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric types (int, float)

    # Check if the target_column is numeric
    if target_column not in numeric_df.columns:
        raise ValueError(f"The target column '{target_column}' must be numeric for correlation analysis.")

    # Calculate correlations on numeric columns only
    correlation_matrix = numeric_df.corr()

    # Select features with correlation to the target variable above the threshold
    correlated_features = correlation_matrix.index[abs(correlation_matrix[target_column]) >= correlation_threshold]
    df_selected = numeric_df[correlated_features]

    print("Columns selected based on correlation threshold of", correlation_threshold, ":", list(df_selected.columns))

    # Optionally display the heatmap
    if plot_heatmap:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_selected.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f"Correlation Heatmap of Features with {target_column}")
        plt.show()

    # Optionally generate scatter plots for each feature against the target variable
    if plot_scatter:
        for feature in df_selected.columns:
            if feature != target_column:
                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=df_selected, x=feature, y=target_column)
                plt.title(f"Scatter Plot of {feature} vs {target_column}")
                plt.xlabel(feature)
                plt.ylabel(target_column)
                plt.show()

    # Return the selected DataFrame for further analysis or plotting
    return df_selected

def create_model_pipeline(model: Any) -> Pipeline:
    """
    Create a machine learning pipeline with standardization.
    
    Args:
        model: A scikit-learn model object
    
    Returns:
        Pipeline: Scikit-learn pipeline with standardization and model
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
def create_model_pipeline(model: Any) -> Pipeline:
    """
    Create a machine learning pipeline with standardization.
    
    Args:
        model: A scikit-learn model object
    
    Returns:
        Pipeline: Scikit-learn pipeline with standardization and model
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    model_name: str = 'Model',
    cv_splits: int = 6
) -> Dict[str, Any]:
    """
    Train and evaluate a machine learning model with cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Scikit-learn model object
        model_name: Name of the model for logging
        cv_splits: Number of cross-validation splits
    
    Returns:
        Dict containing model and evaluation metrics
    """
    print(f"\nTraining and evaluating {model_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = create_model_pipeline(model)
    
    # Cross-validation
    print("Performing cross-validation...")
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Generate predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'model': pipeline,
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2_train': r2_score(y_train, y_train_pred),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2_test': r2_score(y_test, y_test_pred),
        'cv_scores': cv_results
    }
    
    # Print results
    print(f"{model_name} Performance:")
    print(f"  Cross-validation scores: {cv_results}")
    print(f"  Mean CV Score: {cv_results.mean():.4f} (+/- {cv_results.std() * 2:.4f})")
    print(f"  RMSE on training data: {metrics['rmse_train']:.4f}")
    print(f"  R² Score on training data: {metrics['r2_train']:.4f}")
    print(f"  RMSE on test data: {metrics['rmse_test']:.4f}")
    print(f"  R² Score on test data: {metrics['r2_test']:.4f}\n")
    
    return metrics

def hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    param_distributions: Dict[str, Any],
    model_name: str 
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        X: Feature matrix for training
        y: Target vector for training
        model: Scikit-learn model object
        param_distributions: Dictionary of parameter distributions for RandomizedSearchCV
        X_test: Feature matrix for testing
        y_test: Target vector for testing
        model_name: Name of the model for logging
    
    Returns:
        Dict containing best model and evaluation metrics
    """
    print(f"\nStarting hyperparameter tuning for {model_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = create_model_pipeline(model)
    
    rscv = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=KFold(n_splits=6, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rscv.fit(X, y)
    
    # Evaluate best model
    y_train_pred = rscv.best_estimator_.predict(X)
    y_test_pred = rscv.best_estimator_.predict(X_test)
    
    results = {
        'best_model': rscv.best_estimator_,
        'best_params': rscv.best_params_,
        'best_score': rscv.best_score_,
        'rmse_best_train': np.sqrt(mean_squared_error(y, y_train_pred)),
        'r2_best_train': r2_score(y, y_train_pred),
        'rmse_best_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2_best_test': r2_score(y_test, y_test_pred)
    }
    
    print(f"{model_name} Hyperparameter Tuning Results:")
    print(f"  Best Parameters: {results['best_params']}")
    print(f"  Best Cross-Validation Score: {results['best_score']:.4f}")
    print("\nImprovement after hyperparameter tuning:")
    print(f"  RMSE with best model on training data: {results['rmse_best_train']:.4f}")
    print(f"  R² Score with best model on training data: {results['r2_best_train']:.4f}")
    print(f"  RMSE with best model on test data: {results['rmse_best_test']:.4f}")
    print(f"  R² Score with best model on test data: {results['r2_best_test']:.4f}\n")
    
    return results

def main():
    
    #file path
    file_path = r'output\my_data_step5.csv'
    
    # load the data
    data = load_dataframe(file_path = file_path)
    
    # Analyze correlation
    df_selected = analyze_correlation(data, target_column='Price', correlation_threshold=0.2, plot_scatter=False, columns_to_exclude=['Postal_code'])
    
    # Define Models
    models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(eval_metric='rmse')
    }

    # Define hyperparameter distributions
    param_distributions = {
    'Linear Regression': {'model__fit_intercept': [True, False]},
    'Ridge Regression': {'model__alpha': np.linspace(0.0001,1,10), 'model__solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga']},
    'Lasso Regression': {'model__alpha': np.logspace(-3, 3, 7),'model__fit_intercept': [True, False],'model__selection': ['random', 'cyclic']},
    'Random Forest': {'model__n_estimators': [50, 100, 200, 500],'model__max_depth': [None, 10, 20, 30, 40],'model__min_samples_split': [2, 5, 10],'model__min_samples_leaf': [1, 2, 4],'model__max_features': ['sqrt', 'log2'],'model__bootstrap': [True, False]},
    'XGBoost': {'model__n_estimators': [50, 100],'model__max_depth': [3, 5],'model__learning_rate': [0.1, 0.2],'model__subsample': [0.5, 0.75],'model__colsample_bytree': [0.5, 0.75]}
    }

    # Split data into features and target
    X = df_selected.drop(columns=['Price'])  # Assuming 'Price' is the target
    y = df_selected['Price']

    # Train and evaluate each model
    model_results = {}

    for model_name, model in models.items():
        # Train and evaluate the model
        metrics = train_and_evaluate_model(X, y, model, model_name)
        model_results[model_name] = metrics

    # Optionally, perform hyperparameter tuning for each model
    for model_name, model in models.items():
        if model_name in param_distributions:  # Ensure there are parameters defined for tuning
            results = hyperparameter_tuning(X, y, model, param_distributions[model_name], model_name=model_name)
            model_results[f"{model_name} Tuning"] = results

    # Output all model results
    for name, result in model_results.items():
        print(f"\n{name} Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
    # Find the best model based on cross-validation score after tuning
    best_model_name, best_model_result = None, None
    best_cv_score = -np.inf

    # Find the best model based on tuning cross-validation scores
    for name, result in model_results.items():
        if "Tuning" in name:  # Only consider tuned models
            if result['best_score'] > best_cv_score:
                best_cv_score = result['best_score']
                best_model_name = name
                best_model_result = result

    # Display the best model found
    if best_model_result:
        print(f"\nOverall best Model: {best_model_name}")
        print("Best Parameters:", best_model_result['best_params'])
        print("Best Cross-Validation Score:", best_cv_score)

        # Make a sample prediction
        # Select a sample row for comparison
        sample_index = random.randint(0, len(X) - 1) 
        sample_input = X.iloc[sample_index].values.reshape(1, -1)  # Reshape for single prediction

        # Predict the price using the best model
        best_model = best_model_result['best_model']
        sample_prediction = best_model.predict(sample_input)

        # Get the actual price for comparison
        real_price = y.iloc[sample_index]

        # Print results
        print(f"\nSample Prediction with {best_model_name}:")
        print("  Input features:", X.iloc[sample_index].to_dict())
        print("  Predicted Price:", sample_prediction[0])
        print("  Actual Price:", real_price)
        print(f"  Difference (Predicted - Actual): {sample_prediction[0] - real_price:.2f}")
    else:
        print("No tuned models were found.")


    # Plot feature weights or importance for the best model
    def plot_feature_importance(best_model, feature_names):
        if hasattr(best_model.named_steps['model'], 'coef_'):
            # For linear models with coefficients
            weights = best_model.named_steps['model'].coef_
            title = "Feature Weights (Coefficients)"
        elif hasattr(best_model.named_steps['model'], 'feature_importances_'):
            # For ensemble models like Random Forest
            weights = best_model.named_steps['model'].feature_importances_
            title = f"Feature Importance of {best_model_name} "
        else:
            print("The best model does not have feature weights or importance values.")
            return

        # Plot the feature weights or importance
        plt.figure(figsize=(14, max(8, len(feature_names) * 0.35)))
        plt.barh(feature_names, weights, color='skyblue')
        plt.xlabel("Weight / Importance")
        plt.title(title)
        plt.gca().invert_yaxis()  # Reverse order for better visibility
        plt.show()

    # Get feature names from the dataset
    feature_names = X.columns

    # Plot feature weights for the best model if it was found
    if best_model_result:
        plot_feature_importance(best_model, feature_names)
    else:
        print("No best model was found to plot feature weights.")
        

    # Retrain the best model on the entire dataset
    if best_model_result:
        best_model = best_model_result['best_model']
        best_model.fit(X, y)  # Fit the best model on the entire dataset

        # Save the retrained model
        model_filename = 'best_trained_model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"Best model retrained and saved as {model_filename}")
    
    else:
        print("No best model was found to retrain.")
        
    # Save the modified dataframe to CSV
    save_dataframe(data, 'output/my_data_step6')
    
if __name__ == "__main__":
    main()
    
    
# #{'Number_of_bedrooms': 3.0, 'Living_area': 125.0, 'Swimming_Pool': 0.0, 'Number_of_facades': 4.0, 'landSurface': 767.0, 'Has_Assigned_City_15': 0.0, 'State_of_building_encoded': 357414.4556451613, 'epc_encoded': 380425.8285077951}
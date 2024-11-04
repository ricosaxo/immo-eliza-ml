import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import uniform

def load_data(input_path):
    """
    Load data from pickle file and perform initial preprocessing
    
    Args:
        input_path (str): Path to the pickle file
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load pickle file
    df = pd.read_pickle(input_path)
    
    # Select relevant columns
    selected_columns = [
        'Price', 'Number_of_bedrooms', 'Living_area', 'Number_of_facades', 
        'State_of_building', 'epc', 'landSurface', 'Has_Assigned_City', 'Province'
    ]
    df = df[selected_columns].copy(deep=True)
    
    return df

def preprocess_data(df):
    """
    Preprocess the dataframe by encoding categorical variables
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Separate categorical columns
    categorical_df = df.select_dtypes(include=['category', 'object'])
    
    # One-hot encode Province
    province_dummies = pd.get_dummies(categorical_df['Province'], drop_first=True)
    df = pd.concat([df, province_dummies], axis=1)
    df = df.drop('Province', axis=1)
    
    # Ordinal encode EPC
    epc_unique = sorted(categorical_df['epc'].unique(), reverse=True)
    epc_encoder = OrdinalEncoder(categories=[epc_unique])
    df['Encoded_epc'] = epc_encoder.fit_transform(categorical_df[['epc']])
    df = df.drop('epc', axis=1)
    
    # Ordinal encode State of Building
    state_order = ['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
    state_encoder = OrdinalEncoder(categories=[state_order])
    df['Encoded_state_of_building'] = state_encoder.fit_transform(categorical_df[['State_of_building']])
    df = df.drop('State_of_building', axis=1)
    
    return df

def create_model_pipeline(model, scaler=None):
    """
    Create a machine learning pipeline with optional scaling
    
    Args:
        model: Scikit-learn model
        scaler: Scikit-learn scaler (optional)
    
    Returns:
        Pipeline: Scikit-learn pipeline
    """
    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    steps.append(('model', model))
    return Pipeline(steps)

def train_and_evaluate_model(X, y, model, model_name='Model', cv_splits=6):
    """
    Train and evaluate a machine learning model
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target variable
        model: Scikit-learn model
        model_name (str): Name of the model for logging
        cv_splits (int): Number of cross-validation splits
    
    Returns:
        dict: Model performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline with standard scaling
    pipeline = create_model_pipeline(model, StandardScaler())
    
    # Cross-validation
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf)
    
    # Fit model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"Cross-validation scores: {cv_results}")
    print(f"Mean CV Score: {cv_results.mean():.4f} (+/- {cv_results.std() * 2:.4f})")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")
    
    return {
        'model': pipeline,
        'rmse': rmse,
        'r2': r2,
        'cv_scores': cv_results
    }

def hyperparameter_tuning(X, y, model, param_distributions, model_name='Model'):
    """
    Perform hyperparameter tuning using RandomizedSearchCV
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target variable
        model: Scikit-learn model
        param_distributions (dict): Hyperparameter distributions
        model_name (str): Name of the model for logging
    
    Returns:
        dict: Best model and its performance
    """
    # Cross-validation setup
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
    # Create pipeline
    pipeline = create_model_pipeline(model, StandardScaler())
    
    # Randomized Search
    rscv = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_distributions, 
        cv=kf, 
        n_iter=10, 
        random_state=42
    )
    
    rscv.fit(X, y)
    
    print(f"{model_name} Hyperparameter Tuning:")
    print(f"Best Parameters: {rscv.best_params_}")
    print(f"Best Cross-Validation Score: {rscv.best_score_:.4f}\n")
    
    return {
        'best_model': rscv.best_estimator_,
        'best_params': rscv.best_params_,
        'best_score': rscv.best_score_
    }

def main():
    # File paths
    input_pkl = r'..\data\clean\after_step_3b_outliers_cat.pkl'
    output_csv = r'..\data\clean\model_training.csv'
    output_pkl = r'..\data\clean\model_training.pkl'
    
    # Load and preprocess data
    df = load_data(input_pkl)
    df = preprocess_data(df)
    
    # Separate features and target
    X = df.drop('Price', axis=1)
    y = df['Price'].values
    
    # Define models and their hyperparameter distributions
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'model__alpha': np.linspace(0.0001, 1, 10)}),
        'Lasso': (Lasso(), {'model__alpha': np.linspace(0.0001, 1, 10)}),
        'ElasticNet': (
            ElasticNet(), 
            {
                'model__alpha': uniform(0.01, 10),
                'model__l1_ratio': uniform(0, 1)
            }
        ),
        'Random Forest': (
            RandomForestRegressor(random_state=42, oob_score=True), 
            {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30]
            }
        )
    }
    
    # Train and evaluate models
    results = {}
    for name, (model, param_dist) in models.items():
        print(f"Training {name}...")
        results[name] = train_and_evaluate_model(X, y, model, name)
        
        # Perform hyperparameter tuning if distributions are provided
        if param_dist:
            results[name]['tuned'] = hyperparameter_tuning(X, y, model, param_dist, name)
    
    # Save processed data
    df.to_csv(output_csv, index=False)
    with open(output_pkl, 'wb') as f:
        pickle.dump(df, f)
    
    return results

if __name__ == "__main__":
    results = main()
    
    
    
    
    
    
    
    
    
    
    
    import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import uniform

def load_data(input_path):
    """
    Load data from pickle file and perform initial preprocessing
    
    Args:
        input_path (str): Path to the pickle file
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("Loading data from:", input_path)
    pd.set_option('display.max_columns', None)
    df = pd.read_pickle(input_path)
    
    print("Data loaded successfully. Selecting relevant columns.")
    selected_columns = [
        'Price', 'Number_of_bedrooms', 'Living_area', 'Number_of_facades', 
        'State_of_building', 'epc', 'landSurface', 'Has_Assigned_City', 'Province'
    ]
    df = df[selected_columns].copy(deep=True)
    
    print("Columns selected:", selected_columns)
    return df

def preprocess_data(df):
    """
    Preprocess the dataframe by encoding categorical variables
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("Starting data preprocessing.")
    
    # One-hot encode Province
    print("One-hot encoding 'Province' column...")
    province_dummies = pd.get_dummies(df['Province'], drop_first=True)
    df = pd.concat([df, province_dummies], axis=1)
    df = df.drop('Province', axis=1)
    
    # Ordinal encode EPC
    print("Ordinal encoding 'epc' column...")
    epc_unique = sorted(df['epc'].unique(), reverse=True)
    epc_encoder = OrdinalEncoder(categories=[epc_unique])
    df['Encoded_epc'] = epc_encoder.fit_transform(df[['epc']])
    df = df.drop('epc', axis=1)
    
    # Ordinal encode State of Building
    print("Ordinal encoding 'State_of_building' column...")
    state_order = ['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
    state_encoder = OrdinalEncoder(categories=[state_order])
    df['Encoded_state_of_building'] = state_encoder.fit_transform(df[['State_of_building']])
    df = df.drop('State_of_building', axis=1)
    
    print("Data preprocessing completed.")
    return df

def create_model_pipeline(model, scaler=None):
    """
    Create a machine learning pipeline with optional scaling
    
    Args:
        model: Scikit-learn model
        scaler: Scikit-learn scaler (optional)
    
    Returns:
        Pipeline: Scikit-learn pipeline
    """
    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    steps.append(('model', model))
    return Pipeline(steps)

def train_and_evaluate_model(X, y, model, model_name='Model', cv_splits=6):
    """
    Train and evaluate a machine learning model
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target variable
        model: Scikit-learn model
        model_name (str): Name of the model for logging
        cv_splits (int): Number of cross-validation splits
    
    Returns:
        dict: Model performance metrics
    """
    print(f"\nTraining and evaluating {model_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = create_model_pipeline(model, StandardScaler())
    
    # Cross-validation
    print("Performing cross-validation...")
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf)
    
    # Fit model
    print("Fitting the model on the training data.")
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"  Cross-validation scores: {cv_results}")
    print(f"  Mean CV Score: {cv_results.mean():.4f} (+/- {cv_results.std() * 2:.4f})")
    print(f"  RMSE on test data: {rmse:.4f}")
    print(f"  R² Score on test data: {r2:.4f}\n")
    
    return {
        'model': pipeline,
        'rmse': rmse,
        'r2': r2,
        'cv_scores': cv_results
    }

def hyperparameter_tuning(X, y, model, param_distributions, model_name='Model'):
    """
    Perform hyperparameter tuning using RandomizedSearchCV
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target variable
        model: Scikit-learn model
        param_distributions (dict): Hyperparameter distributions
        model_name (str): Name of the model for logging
    
    Returns:
        dict: Best model and its performance
    """
    print(f"\nStarting hyperparameter tuning for {model_name}...")
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
    pipeline = create_model_pipeline(model, StandardScaler())
    
    rscv = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_distributions, 
        cv=kf, 
        n_iter=10, 
        random_state=42
    )
    
    rscv.fit(X, y)
    
    print(f"{model_name} Hyperparameter Tuning Results:")
    print(f"  Best Parameters: {rscv.best_params_}")
    print(f"  Best Cross-Validation Score: {rscv.best_score_:.4f}\n")
    
    return {
        'best_model': rscv.best_estimator_,
        'best_params': rscv.best_params_,
        'best_score': rscv.best_score_
    }

def main():
    input_pkl = r'..\data\clean\after_step_3b_outliers_cat.pkl'
    output_csv = r'..\data\clean\model_training.csv'
    output_pkl = r'..\data\clean\model_training.pkl'
    
    print("Loading and preprocessing data...")
    df = load_data(input_pkl)
    df = preprocess_data(df)
    
    # Separate features and target
    X = df.drop('Price', axis=1)
    y = df['Price'].values
    
    # Define models and their hyperparameter distributions
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'model__alpha': np.linspace(0.0001, 1, 10)}),
        'Lasso': (Lasso(), {'model__alpha': np.linspace(0.0001, 1, 10)}),
        'ElasticNet': (
            ElasticNet(), 
            {
                'model__alpha': uniform(0.01, 10),
                'model__l1_ratio': uniform(0, 1)
            }
        ),
        'Random Forest': (
            RandomForestRegressor(random_state=42, oob_score=True), 
            {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30]
            }
        )
    }
    
    # Train and evaluate models
    results = {}
    for name, (model, param_dist) in models.items():
        print(f"Training {name}...")
        results[name] = train_and_evaluate_model(X, y, model, name)
        
        # Perform hyperparameter tuning if distributions are provided
        if param_dist:
            print(f"Tuning hyperparameters for {name}...")
            results[name]['tuned'] = hyperparameter_tuning(X, y, model, param_dist, name)
    
    print("Saving processed data and results...")
    df.to_csv(output_csv, index=False)
    with open(output_pkl, 'wb') as f:
        pickle.dump(df, f)
    
    print("Training and evaluation complete. Results:")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}")
        if 'tuned' in result:
            print(f"  Best Params after tuning: {result['tuned']['best_params']}")
            print(f"  Best CV Score after tuning: {result['tuned']['best_score']:.4f}")
    return results

if __name__ == "__main__":
    results = main()

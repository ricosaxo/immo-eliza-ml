print('hello')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Example data, replace X and y with your dataset if available
X, y = make_regression(n_samples=500, n_features=5, noise=0.3, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with alpha range for Ridge and Lasso
models = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge Regression": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso Regression": (Lasso(), {"alpha": [0.01, 0.1, 1, 10, 100]})
}

# Store cross-validation results
results = []
kf = KFold(n_splits=6, random_state=42, shuffle=True)

# Evaluate each model with cross-validation and tuning
for model_name, (model, params) in models.items():
    if params:  # If there are hyperparameters to tune
        grid_search = GridSearchCV(model, params, cv=kf, scoring="neg_mean_squared_error")
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        print(f"Best alpha for {model_name}: {grid_search.best_params_['alpha']}")
    else:
        # For Linear Regression (no alpha), perform cross-validation directly
        best_score = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="neg_mean_squared_error").mean()
        best_model = model
    results.append(best_score)

# Plot the cross-validation MSE results for each model
plt.bar(models.keys(), results)
plt.ylabel("Average Negative MSE (Higher is Better)")
plt.title("Cross-Validation Results for Linear Regression Models with Alpha Tuning")
plt.show()

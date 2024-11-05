# Immo Eliza ML

## Overview

Immo Eliza ML is a machine learning project aimed at predicting real estate prices in Belgium. Using various regression models, this project preprocesses a dataset of real estate properties to create a predictive model that estimates prices based on numerous features.

## Learning Objectives

The main objectives of this project include:

- Preprocessing data for machine learning.
- Applying linear regression in a real-life context.
- Exploring various machine learning models for regression tasks.
- Evaluating model performance using appropriate metrics.
- (Optional) Implementing hyperparameter tuning and cross-validation techniques.

## Dataset

The dataset consists of approximately 4180 properties scraped in Belgium of immoweb ([www.immoweb.be](https://www.immoweb.be/)) in october 2024, and contains mainly houses.

The dataset used for this project is located in the 'data\clean\after_step_4_correlation.pkl' file, but it is also possible to load a preprocessed dataset from a pickle file.

## Installation

To set up the project, ensure you have Python 3.6 or higher installed, and follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/immo-eliza-ml.git
   cd immo-eliza-ml

   ```

2. Create a virtual environment:

python -m venv venv

3. Activate the virtual environment:

- On windows:

venv\Scripts\activate

- On macOS/Linux:

source venv/bin/activate

4. Install the required packages:

pip install -r requirements.txt

## Project Overview

This project loads a dataset containing various features related to real estate properties and their corresponding prices. It performs the following tasks:

Data Loading and Preprocessing: Loads data from a pickle file, checks for missing values, and drops irrelevant columns.

Exploratory Data Analysis (EDA): Generates correlation heatmaps and scatter plots for visual analysis of feature relationships.

Model Training and Evaluation: Implements multiple regression models, evaluates their performance using cross-validation, and reports metrics such as RMSE and R² scores.

Hyperparameter Tuning: Utilizes RandomizedSearchCV to optimize model parameters for better performance.

Feature Importance Visualization: Plots feature weights or importance for the best-performing model.

Model Saving: Saves the retrained best model for future predictions

## Requirements:

To run this project, ensure you have the following libraries installed:

pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

joblib

You can install the required packages using pip:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib

## Data

The data should be in a pickle format. The script expects a file path to a cleaned dataset that has undergone preprocessing.

Modify the input_path variable in the script to point to your dataset:

input_path = r'data\clean\after_step_4_correlation.pkl'

## Usage

1. Clone this repository to your local machine:

git clone <repository-url>

2.Navigate to the project directory:

cd <project-directory>

3. Run the script:

python your_script.py

## Model Evaluation

The following metrics are calculated during the evaluation phase:

RMSE (Root Mean Squared Error): Measures the average magnitude of the errors between predicted and actual values.

R² Score: Represents the proportion of variance in the target variable that can be explained by the features.

## Hyperparameter Tuning

For each model, hyperparameter tuning is performed using RandomizedSearchCV, optimizing parameters such as regularization strength for regression models and tree-based parameters for ensemble methods.

## Feature Importance

The script visualizes feature importance for the best-performing model. This helps in understanding which features contribute most to price prediction.

## Saving the Model

The best-trained model is retrained on the entire dataset and saved as a .pkl file using joblib. This allows for future use without retraining the model.

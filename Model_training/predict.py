import joblib
import pandas as pd
import numpy as np

def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("Model file not found.")
        return None

def predict_price(model, input_data: pd.DataFrame):
    prediction = model.predict(input_data)
    return prediction

def main():
    model_path = 'best_trained_model.pkl'
    model = load_model(model_path)
    
    if model:
        # Example input: replace with actual data input
        sample_data = pd.DataFrame([{
            # Replace with actual feature values
            # Add all relevant features as per your model's requirements
        }])
        
        prediction = predict_price(model, sample_data)
        print("Predicted Price:", prediction[0])

if __name__ == '__main__':
    main()
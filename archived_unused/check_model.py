import joblib
from pathlib import Path

model_path = Path("modeling/models/pollutant_aqi_regression_xgboost.pkl")

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

print(f"\nModel type: {type(model)}")
print(f"Model attributes: {dir(model)}")

if hasattr(model, 'predict'):
    print("\n Model has 'predict' method")
else:
    print("\n Model DOES NOT have 'predict' method")
    print(f"\nActual model content:")
    print(model)
    
    if isinstance(model, dict):
        print("\nDict keys:", model.keys())
        if 'model' in model:
            print(f"\nExtracted model type: {type(model['model'])}")
            print(f"Extracted model has predict: {hasattr(model['model'], 'predict')}")

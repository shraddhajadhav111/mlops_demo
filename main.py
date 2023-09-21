from src.data_fetcher import load_data
from src.data_preprocessor import select_and_preprocess_data
from src.model import main as model_main

def run_pipeline():
    # Load the data
    df = load_data(r"C:\Users\shjadhav\OneDrive - Malomatia\Desktop\ml_cicd\data\train.csv")

    # Preprocess the data
    X_train, X_test, y_train, y_test = select_and_preprocess_data(df)

    # Train the model, evaluate it, and log results with mlflow
    results = model_main(X_train, y_train, X_test, y_test)

    print(results)

if __name__ == "__main__":
    run_pipeline()
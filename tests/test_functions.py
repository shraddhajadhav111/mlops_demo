import unittest
from src.data_fetcher import load_data
from src.data_preprocessor import select_and_preprocess_data
from src.model import train_model, predict, evaluation, rmse_cv
import mlflow

class TestModel(unittest.TestCase):
    
    def setUp(self):
        self.df = load_data("data/train.csv")
        self.X_train, self.X_test, self.y_train, self.y_test = select_and_preprocess_data(self.df)

    def test_train_model(self):
        with mlflow.start_run(run_name="Test Train Model"):
            model = train_model(self.X_train, self.y_train)
            self.assertIsNotNone(model)
            mlflow.log_param("Training Samples", len(self.X_train))
            mlflow.xgboost.log_model(model, "Test Train Model")

    def test_predictions(self):
        with mlflow.start_run(run_name="Test Predictions"):
            model = train_model(self.X_train, self.y_train)
            predictions = predict(model, self.X_test)
            self.assertEqual(len(predictions), len(self.y_test))
            mlflow.log_metric("Number of Predictions", len(predictions))

    def test_evaluation(self):
        with mlflow.start_run(run_name="Test Evaluation"):
            model = train_model(self.X_train, self.y_train)
            predictions = predict(model, self.X_test)
            mae, mse, rmse, r_squared = evaluation(self.y_test, predictions)
            self.assertGreaterEqual(r_squared, 0)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2 Score", r_squared)

    def test_rmse_cv(self):
        with mlflow.start_run(run_name="Test RMSE CV"):
            model = train_model(self.X_train, self.y_train)
            rmse = rmse_cv(model, self.X_train, self.y_train)
            self.assertGreaterEqual(rmse, 0)
            mlflow.log_metric("RMSE Cross-Validation", rmse)

if __name__ == "__main__":
    unittest.main()

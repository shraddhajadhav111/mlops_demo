import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train, n_estimators=1000, learning_rate=0.01):
    """
    Train an XGBRegressor model.
    """
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """
    Predict using the trained model.
    """
    return model.predict(X_test)

def evaluation(y_true, y_pred):
    """
    Calculate evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def rmse_cv(model, X, y, cv=5):
    """
    Calculate RMSE using cross-validation.
    """
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv))
    return rmse.mean()

def main(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="XGBRegressor Training"):
        model = train_model(X_train, y_train)
        predictions = predict(model, X_test)
        mae, mse, rmse, r_squared = evaluation(y_test, predictions)
        
        # Log metrics to mlflow
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2 Score", r_squared)
        
        rmse_cross_val = rmse_cv(model, X_train, y_train)
        mlflow.log_metric("RMSE Cross-Validation", rmse_cross_val)
        
        # Log model to mlflow
        mlflow.xgboost.log_model(model, "model")
        # Register model to mlflow
        model_uri = "runs:/{run_id}/model".format(run_id=mlflow.active_run().info.run_id)
        registered_model_name = "XGBRegressorModel"
        
    return mae, mse, rmse, r_squared, rmse_cross_val

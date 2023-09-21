import pandas as pd
import logging
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def select_and_preprocess_data(df):
    """
    Select specified numerical and categorical columns, perform one-hot encoding, and split the data.
    
    Parameters:
    - df: Input DataFrame
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    with mlflow.start_run(run_name="Data Preprocessing"):
        # Specified numerical columns
        num_cols = ['Id', 'MSSubClass', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                    'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']

        # Specified categorical columns
        cat_cols = ["MSZoning", "Utilities", "BldgType", "Heating", "KitchenQual", "SaleCondition", "LandSlope"]

        # Check if columns are missing in the DataFrame
        missing_cols = [col for col in num_cols + cat_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"The following columns are missing in the DataFrame: {missing_cols}")
            mlflow.log_param("Error", "Missing columns in DataFrame")
            return None, None, None, None

        # Select specified columns and drop the rest
        df = df[num_cols + cat_cols]

        # One-hot encoding for categorical columns
        df = pd.get_dummies(df, columns=cat_cols)

        # Split data
        X = df.drop("SalePrice", axis=1)
        y = df["SalePrice"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Log parameters and metrics to mlflow
        mlflow.log_param("Number of Features", X_train.shape[1])
        mlflow.log_param("Number of Training Samples", X_train.shape[0])
        mlflow.log_param("Number of Testing Samples", X_test.shape[0])

    return X_train, X_test, y_train, y_test

# Example usage:
# df = pd.read_csv("path_to_data.csv")
# X_train, X_test, y_train, y_test = select_and_preprocess_data(df)

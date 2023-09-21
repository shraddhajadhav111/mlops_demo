import pandas as pd 
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """
    Load data from a given path.
    
    Parameters:
    - path: Path to the data file
    
    Returns:
    - Loaded DataFrame
    """
    return pd.read_csv(path)


import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path):
    """
    Load data from a given path.
    
    Parameters:
    - path: Path to the data file
    
    Returns:
    - Loaded DataFrame or None if an error occurs
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}")
        logging.info(f"Columns in loaded data: {df.columns}")

        return df
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
    except pd.errors.EmptyDataError:
        logger.error(f"No data in file at {path}")
    except pd.errors.ParserError:
        logger.error(f"Error parsing the file at {path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None

# Example usage:
# df = load_data("/path_to_data/train.csv")  # Adjust path as needed
# if df is not None:
#     # Continue with further processing

import pandas as pd
from config import Config
from data_loader import DataLoader
import numpy as np
from logger import Logger  # Import the Logger class

def main():
    # Set up logger
    logger = Logger.setup_logger()

    try:
        # Reading data from Google Drive
        sample_dataset_path = 'Data/sample_dataset.csv'
        target_column = 'label'
        
        # Log data loading
        logger.info(f"1. Loading dataset from {sample_dataset_path}")
        
        # Load sample data as csv using Pandas library
        df = pd.read_csv(sample_dataset_path)
        logger.info(f"2. Dataset loaded successfully, shape: {df.shape}")
        
        # Print the first 5 rows of the dataset
        logger.info("3. Displaying first 5 rows of the dataset:")
        logger.info("\n" + df.head(5).to_string())
        
        # Initialize DataLoader
        data_loader = DataLoader(dataframe=df, time_steps=Config.N_TIME_STEPS, step=Config.step, target_column=target_column)
        
        # Load data (this will return a DataFrame with segments, labels, eventID, and userID)
        logger.info("4. Loading data with DataLoader...")
        df_sample = data_loader.load_data()
        logger.info(f"5. Data loaded successfully, resulting dataframe shape: {df_sample.shape}")
        
        # Print only the first row of the DataLoader output
        logger.info("6. Displaying first row of processed data:")
        #logger.info("\n" + df_sample.head(1).to_string())  # Print only one row
        logger.info("7. Uncomment above row to view DataLoader output")

        # Indicate pipeline completion
        logger.info("8. Pipeline complete.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

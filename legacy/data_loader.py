import pandas as pd
from typing import List, Tuple

class DataLoader:
    """
    DataLoader class for loading, cleaning, and splitting financial data from CSV.
    Follows SOLID principles and clean architecture for the persistence layer.
    """
    
    # Class variables
    # FILE_PATH: str = "path/to/your/data.csv"  # Pre-defined path to CSV file
    FILE_PATH: str = "/Users/jack/Projects/lstm/forecasting/data/raw/gold_1minute.csv"  # Pre-defined path to CSV file
    SELECTED_COLUMNS: List[str] = ['close']  # Columns to select
    SPLIT_RATIOS: List[float] = [0.95, 0.035, 0.015]  # Train, validation, test ratios
    
    def __init__(self):
        self._raw_data: pd.DataFrame = self._load_data()
        self._cleaned_data: pd.DataFrame = self._clean_data(self._raw_data)
        self._ordered_data: pd.DataFrame = self._order_data(self._cleaned_data)
        self._processed_data: pd.DataFrame = self._remove_gaps(self._ordered_data)
        self._train_data, self._val_data, self._test_data = self._split_data(self._processed_data)
    
    def _load_data(self) -> pd.DataFrame:
        """Private method to load CSV and set date/time as index."""
        df = pd.read_csv(self.FILE_PATH, header=0)
        df.set_index(df.columns[0], inplace=True)  # First column as index
        df.index = pd.to_datetime(df.index)  # Ensure datetime index
        return df[self.SELECTED_COLUMNS]  # Select specified columns
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Private method for data cleaning: handle missing values."""
        # Drop rows where all values are NaN
        return df.dropna(how='all')
    
    def _order_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Private method to order data by index."""
        return df.sort_index()
    
    def _remove_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Private method to remove gaps: forward fill missing values."""
        return df.ffill()
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Private method to split data into train, validation, test."""
        n = len(df)
        train_end = int(n * self.SPLIT_RATIOS[0])
        val_end = train_end + int(n * self.SPLIT_RATIOS[1])
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        return train, val, test
    
    # Getters
    def get_train_data(self) -> pd.DataFrame:
        """Get training data."""
        return self._train_data
    
    def get_validation_data(self) -> pd.DataFrame:
        """Get validation data."""
        return self._val_data
    
    def get_test_data(self) -> pd.DataFrame:
        """Get test data."""
        return self._test_data
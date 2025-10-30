"""CSV data source adapter."""
import pandas as pd

from domain.ports import DataSource


class CSVDataSource(DataSource):
    """Loads time series data from CSV file."""

    def __init__(self, filepath: str, target_column: str):
        """Initialize with file path and target column."""
        self.filepath = filepath
        self.target_column = target_column

    def load(self) -> pd.DataFrame:
        """Load CSV file."""
        df = pd.read_csv(self.filepath, header=0)
        df.set_index(df.columns[0], inplace=True)
        df.index = pd.to_datetime(df.index)
        return df[[self.target_column]]

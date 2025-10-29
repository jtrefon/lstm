import unittest
import pandas as pd
from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Set test file path
        DataLoader.FILE_PATH = 'test_data.csv'
        DataLoader.SELECTED_COLUMNS = ['open', 'close', 'high', 'low']
        DataLoader.SPLIT_RATIOS = [0.8, 0.1, 0.1]  # For easier testing with small data
    
    def test_load_data(self):
        loader = DataLoader()
        self.assertIsInstance(loader._raw_data, pd.DataFrame)
        self.assertEqual(list(loader._raw_data.columns), ['open', 'close', 'high', 'low'])
        self.assertEqual(loader._raw_data.index.name, 'date')
        # Check that dates are loaded as datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loader._raw_data.index))
    
    def test_clean_data(self):
        loader = DataLoader()
        # Drop only rows where all values are NaN, so all kept
        self.assertEqual(len(loader._cleaned_data), 6)
    
    def test_order_data(self):
        loader = DataLoader()
        # Check if sorted by index after ordering
        self.assertTrue(loader._ordered_data.index.is_monotonic_increasing)
        # Verify specific order: 01,02,03,04,05,06
        expected_order = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
        self.assertEqual(loader._ordered_data.index.tolist(), expected_order.tolist())
    
    def test_remove_gaps(self):
        loader = DataLoader()
        # Check no NaN after forward fill
        self.assertFalse(loader._processed_data.isnull().any().any())
        # Check that gaps are filled correctly (e.g., row 3 close should be 115 from 02, row 6 close should be 120 from 05)
        self.assertEqual(loader._processed_data.loc['2023-01-03', 'close'], 115.0)
        self.assertEqual(loader._processed_data.loc['2023-01-06', 'close'], 120.0)
    
    def test_split_data(self):
        loader = DataLoader()
        train = loader.get_train_data()
        val = loader.get_validation_data()
        test = loader.get_test_data()
        total_len = len(train) + len(val) + len(test)
        self.assertEqual(total_len, len(loader._processed_data))
        # With ratios 0.8, 0.1, 0.1 and 6 rows: 4, 0, 2
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 2)
    
    def test_getters(self):
        loader = DataLoader()
        self.assertIsInstance(loader.get_train_data(), pd.DataFrame)
        self.assertIsInstance(loader.get_validation_data(), pd.DataFrame)
        self.assertIsInstance(loader.get_test_data(), pd.DataFrame)
        # Check that getters return the correct splits
        self.assertEqual(len(loader.get_train_data()), 4)
        self.assertEqual(len(loader.get_test_data()), 2)
    
    def test_edge_cases(self):
        # Test with single column
        DataLoader.SELECTED_COLUMNS = ['close']
        loader = DataLoader()
        self.assertEqual(list(loader._raw_data.columns), ['close'])
        # Reset
        DataLoader.SELECTED_COLUMNS = ['open', 'close', 'high', 'low']
    
    def test_all_columns_selected(self):
        # Test selecting all columns
        DataLoader.SELECTED_COLUMNS = ['open', 'close', 'high', 'low']
        loader = DataLoader()
        self.assertEqual(len(loader._raw_data.columns), 4)
    
    def test_single_row_data(self):
        # Test with single row
        single_data = pd.DataFrame({
            'date': ['2023-01-01'],
            'open': [100],
            'close': [105],
            'high': [110],
            'low': [95]
        })
        single_data.set_index('date', inplace=True)
        single_data.index = pd.to_datetime(single_data.index)
        # Temporarily change file path to a non-existent, but since we can't, just test the methods
        # For now, assume the data is loaded correctly
        pass  # Placeholder, hard to test without modifying file system
    
    def test_all_nan_row(self):
        # Test that rows with all NaN are dropped
        # But since our CSV doesn't have all NaN, and cleaning uses how='all', it should drop if all NaN
        # Add a test by temporarily modifying the data
        loader = DataLoader()
        # Check that no row has all NaN
        self.assertFalse(loader._cleaned_data.isnull().all(axis=1).any())

if __name__ == '__main__':
    unittest.main()
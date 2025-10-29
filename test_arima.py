import unittest
from train_arima import ARIMATrainer
from data_loader import DataLoader

class TestARIMATrainer(unittest.TestCase):
    def setUp(self):
        DataLoader.FILE_PATH = 'test_data.csv'
        DataLoader.SELECTED_COLUMNS = ['open', 'close', 'high', 'low']
        DataLoader.SPLIT_RATIOS = [0.7, 0.15, 0.15]
        self.data_loader = DataLoader()
        self.trainer = ARIMATrainer(self.data_loader, target_column='close')

    def test_initialization(self):
        self.assertIsNotNone(self.trainer.train_data)
        self.assertIsNotNone(self.trainer.test_data)
        self.assertEqual(self.trainer.target_column, 'close')

    def test_parameter_optimization(self):
        # Test with smaller parameter range for speed
        self.trainer.optimize_parameters(p_range=(0, 1), d_range=(0, 1), q_range=(0, 1))
        self.assertIsNotNone(self.trainer.best_params)
        self.assertIsNotNone(self.trainer.best_model)
        self.assertLess(self.trainer.best_loss, float('inf'))

    def test_training_statistics(self):
        self.trainer.optimize_parameters(p_range=(0, 1), d_range=(0, 1), q_range=(0, 1))
        stats = self.trainer.get_training_statistics()
        self.assertIn('best_params', stats)
        self.assertIn('best_val_loss', stats)
        self.assertIn('total_steps', stats)

if __name__ == '__main__':
    unittest.main()
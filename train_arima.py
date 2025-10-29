import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from data_loader import DataLoader
import warnings
warnings.filterwarnings('ignore')

@dataclass(frozen=True)
class ARIMAConfig:
    target_column: str = 'close'
    p_range: Tuple[int, int] = (0, 2)
    d_range: Tuple[int, int] = (0, 1)
    q_range: Tuple[int, int] = (0, 2)
    early_stop_tolerance: int = 5
    optimization_train_window: Optional[int] = 1000
    optimization_val_window: Optional[int] = 100
    plot_max_points: int = 500


@dataclass
class ParameterSearchStep:
    params: Tuple[int, int, int]
    val_loss: float
    aic: float
    bic: float


@dataclass
class ParameterSearchResult:
    best_params: Tuple[int, int, int]
    best_model: ARIMA
    best_loss: float
    steps: List[ParameterSearchStep]
    early_stopped: bool


@dataclass
class ForecastResult:
    forecast: pd.Series
    actual: pd.Series
    mse: float
    rmse: float
    mae: float
    valid_actual: np.ndarray
    valid_forecast: np.ndarray


class ARIMAValidator:
    """Encapsulates walk-forward validation logic for ARIMA.
    
    Performs rolling-window validation where the model is retrained at each step
    to evaluate parameter robustness. This is computationally expensive but provides
    realistic performance estimates for time series data.
    """

    def __init__(self, train_series: pd.Series, validation_series: pd.Series):
        """
        Initialize validator with training and validation series.
        
        Args:
            train_series: Historical data for initial model training
            validation_series: Data for walk-forward evaluation
            
        Raises:
            ValueError: If either series is empty
        """
        if len(train_series) == 0:
            raise ValueError("Training series is empty; cannot validate ARIMA parameters.")
        if len(validation_series) == 0:
            raise ValueError("Validation series is empty; cannot validate ARIMA parameters.")

        self._train_series = train_series
        self._validation_series = validation_series

    @property
    def train_series(self) -> pd.Series:
        """Get the training series."""
        return self._train_series

    def walk_forward_loss(self, order: Tuple[int, int, int]) -> float:
        """
        Compute MSE using walk-forward validation.
        
        For each validation point, fits ARIMA on history and forecasts one step ahead.
        Updates history with actual value and repeats. This simulates real-world deployment.
        
        Args:
            order: ARIMA order tuple (p, d, q)
            
        Returns:
            Mean squared error across all validation steps, or inf if fitting fails
        """
        history = self._train_series.copy()
        forecasts: List[float] = []
        actuals: List[float] = []

        for idx, actual in self._validation_series.items():
            try:
                model = ARIMA(history, order=order)
                fitted = model.fit()
                step_forecast = fitted.forecast(steps=1)
                predicted = step_forecast.iloc[0] if hasattr(step_forecast, 'iloc') else step_forecast[0]
            except Exception as exc:
                print(f"  Walk-forward validation error at {idx}: {exc}")
                return float('inf')

            forecasts.append(predicted)
            actuals.append(actual)
            history = pd.concat([history, pd.Series([actual], index=[idx])])

        if not forecasts:
            return float('inf')

        return mean_squared_error(actuals, forecasts)


class ARIMAOptimizer:
    """Coordinates ARIMA parameter search using a validator.
    
    Performs grid search over ARIMA parameter space with walk-forward validation.
    Supports early stopping when no improvement is observed.
    """

    def __init__(self, validator: ARIMAValidator, config: ARIMAConfig):
        """
        Initialize optimizer.
        
        Args:
            validator: ARIMAValidator instance for evaluating parameters
            config: ARIMAConfig with parameter ranges and optimization settings
        """
        self.validator = validator
        self.config = config

    def _parameter_grid(self) -> Iterable[Tuple[int, int, int]]:
        """Generate all ARIMA parameter combinations from config ranges."""
        p_values = range(self.config.p_range[0], self.config.p_range[1] + 1)
        d_values = range(self.config.d_range[0], self.config.d_range[1] + 1)
        q_values = range(self.config.q_range[0], self.config.q_range[1] + 1)
        return product(p_values, d_values, q_values)

    def run(self) -> ParameterSearchResult:
        """
        Execute grid search over ARIMA parameters.
        
        Returns:
            ParameterSearchResult with best parameters, model, and search history
            
        Raises:
            RuntimeError: If no viable parameters are found
        """
        print("Starting ARIMA parameter optimization with walk-forward validation...")

        history: List[ParameterSearchStep] = []
        best_loss = float('inf')
        best_params: Optional[Tuple[int, int, int]] = None
        best_model: Optional[ARIMA] = None
        no_improvement = 0

        grid = list(self._parameter_grid())
        print(f"Total parameter combinations: {len(grid)}")

        for idx, params in enumerate(grid, start=1):
            order = tuple(params)
            try:
                val_loss = self.validator.walk_forward_loss(order)
                fitted_model = ARIMA(self.validator.train_series, order=order).fit()
                step = ParameterSearchStep(
                    params=order,
                    val_loss=val_loss,
                    aic=fitted_model.aic,
                    bic=fitted_model.bic,
                )
                history.append(step)
                print(
                    f"Step {idx}/{len(grid)}: ARIMA{order} - Val Loss: {val_loss:.4f}, "
                    f"AIC: {step.aic:.2f}, BIC: {step.bic:.2f}"
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = order
                    best_model = fitted_model
                    no_improvement = 0
                    print(f"  -> New best model! Loss: {val_loss:.4f}")
                else:
                    no_improvement += 1
                    print(f"  -> No improvement. Count: {no_improvement}")

                if no_improvement >= self.config.early_stop_tolerance:
                    print(
                        f"Early stopping at step {idx}. "
                        f"No improvement for {self.config.early_stop_tolerance} iterations."
                    )
                    break

            except Exception as exc:
                print(f"Error with ARIMA{order}: {exc}")

        if best_params is None or best_model is None:
            raise RuntimeError("Parameter optimization failed to identify a viable ARIMA model.")

        print(f"\nOptimization complete. Best parameters: ARIMA{best_params}")
        print(f"Best validation loss: {best_loss:.4f}")

        early_stopped = len(history) < len(grid)
        return ParameterSearchResult(
            best_params=best_params,
            best_model=best_model,
            best_loss=best_loss,
            steps=history,
            early_stopped=early_stopped,
        )


class ARIMAEvaluator:
    """Generates rolling forecasts on a hold-out test split.
    
    Uses walk-forward evaluation to simulate real-world deployment where
    the model is continuously updated with new observations.
    """

    def __init__(
        self,
        train_series: pd.Series,
        validation_series: pd.Series,
        test_series: pd.Series,
        best_params: Tuple[int, int, int],
    ):
        """
        Initialize evaluator.
        
        Args:
            train_series: Training data
            validation_series: Validation data (combined with train for final history)
            test_series: Hold-out test data for evaluation
            best_params: ARIMA order tuple to use for forecasting
            
        Raises:
            ValueError: If best_params is None
        """
        if best_params is None:
            raise ValueError("Best parameters are required for evaluation.")

        self.history = pd.concat([train_series, validation_series]) if len(validation_series) > 0 else train_series
        self.test_series = test_series
        self.best_params = best_params

    def evaluate(self) -> ForecastResult:
        """
        Perform walk-forward evaluation on test set.
        
        For each test point, fits ARIMA model on accumulated history and forecasts
        one step ahead. Actual value is added to history for next iteration.
        
        Returns:
            ForecastResult with predictions, actuals, and performance metrics
        """
        if len(self.test_series) == 0:
            print("No test data available for evaluation.")
            empty = pd.Series(dtype=float)
            return ForecastResult(empty, empty, float('inf'), float('inf'), float('inf'), np.array([]), np.array([]))

        history = self.history.copy()
        predictions: List[float] = []

        total_steps = len(self.test_series)
        print(f"Starting rolling forecast evaluation on {total_steps} test points...")

        for idx, actual in self.test_series.items():
            step_num = len(predictions) + 1
            if step_num % max(1, total_steps // 10) == 0:
                print(f"  Progress: {step_num}/{total_steps} ({100*step_num/total_steps:.1f}%)")

            try:
                fitted = ARIMA(history, order=self.best_params).fit()
                step_forecast = fitted.forecast(steps=1)
                predicted = step_forecast.iloc[0] if hasattr(step_forecast, 'iloc') else step_forecast[0]
            except Exception as exc:
                print(f"  Test evaluation error at {idx}: {exc}")
                predicted = np.nan

            predictions.append(predicted)
            history = pd.concat([history, pd.Series([actual], index=[idx])])

        forecast_series = pd.Series(predictions, index=self.test_series.index)
        actual_series = self.test_series

        valid_mask = ~np.isnan(forecast_series.values) & ~np.isnan(actual_series.values)
        if np.any(valid_mask):
            valid_actual = actual_series.values[valid_mask]
            valid_forecast = forecast_series.values[valid_mask]
            mse = mean_squared_error(valid_actual, valid_forecast)
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(valid_actual - valid_forecast)))
            print("\nTest Set Forecasting Performance:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Evaluated on {len(valid_forecast)} valid samples")
        else:
            print("\nNo valid forecast samples available for evaluation.")
            mse = rmse = mae = float('inf')
            valid_actual = np.array([])
            valid_forecast = np.array([])

        preview_count = min(5, len(actual_series))
        if preview_count > 0:
            comparison = pd.DataFrame(
                {
                    'actual': actual_series.iloc[:preview_count].values,
                    'predicted': forecast_series.iloc[:preview_count].values,
                },
                index=actual_series.index[:preview_count],
            )
            print("\nSample actual vs predicted (first {0} points):".format(preview_count))
            print(comparison)
            print(
                f"Actual range: [{actual_series.min():.2f}, {actual_series.max():.2f}] | "
                f"Predicted range: [{forecast_series.min():.2f}, {forecast_series.max():.2f}]"
            )

        return ForecastResult(
            forecast=forecast_series,
            actual=actual_series,
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            valid_actual=valid_actual,
            valid_forecast=valid_forecast,
        )


class ARIMAPlotter:
    """Responsible for plotting ARIMA forecasts."""

    def __init__(self, max_points: int):
        self.max_points = max_points

    def plot(self, forecast: ForecastResult, target_column: str) -> None:
        if forecast.forecast.empty or forecast.actual.empty:
            print("Forecast or actual data empty. Skipping plot.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting.")
            return

        steps = min(len(forecast.actual), self.max_points)
        forecast_series = forecast.forecast.iloc[:steps]
        actual_series = forecast.actual.iloc[:steps]

        plt.figure(figsize=(12, 5))
        plt.plot(actual_series.index, actual_series.values, label='Actual', color='tab:blue')
        plt.plot(
            forecast_series.index,
            forecast_series.values,
            label='Predicted',
            color='tab:orange',
            linestyle='--',
        )
        plt.title('ARIMA Forecast vs Actual (Test Set)')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class ARIMATrainer:
    """Facade coordinating data preparation, optimization, training, and evaluation."""

    ENABLE_PLOTTING: bool = False

    def __init__(self, data_loader: DataLoader, config: Optional[ARIMAConfig] = None, target_column: Optional[str] = None):
        """
        Initialize ARIMA trainer with data and configuration.
        
        Args:
            data_loader: DataLoader instance providing train/val/test splits
            config: ARIMAConfig instance (uses defaults if None)
            target_column: Override target column from config (for backward compatibility)
        """
        self.config = config or ARIMAConfig()
        if target_column is not None:
            # Support legacy API: ARIMATrainer(loader, target_column='close')
            self.config = ARIMAConfig(target_column=target_column)
        
        self.data_loader = data_loader

        # Extract target column and validate
        self.train_series = data_loader.get_train_data()[self.config.target_column].copy()
        self.val_series = data_loader.get_validation_data()[self.config.target_column].copy()
        self.test_series = data_loader.get_test_data()[self.config.target_column].copy()

        if len(self.train_series) == 0:
            raise ValueError("Training split is empty. Adjust DataLoader.SPLIT_RATIOS or verify source data.")
        # Allow empty validation split for small datasets
        if len(self.val_series) == 0:
            print("Warning: Validation split is empty. Using training data for validation.")

        # Legacy attribute names for backward compatibility
        self.train_data = self.train_series
        self.test_data = self.test_series
        self.target_column = self.config.target_column

        self.optimization_result: Optional[ParameterSearchResult] = None
        self.trained_model: Optional[ARIMA] = None
        self.best_params: Optional[Tuple[int, int, int]] = None
        self.best_loss: Optional[float] = None
        self.best_model: Optional[ARIMA] = None  # Alias for trained_model (backward compatibility)
        self.last_forecast: Optional[ForecastResult] = None

    # --------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------
    def _prepare_optimization_series(self) -> Tuple[pd.Series, pd.Series]:
        """Prepare train and validation series for optimization, applying window limits."""
        train_series = self.train_series
        val_series = self.val_series

        if self.config.optimization_train_window and len(train_series) > self.config.optimization_train_window:
            train_series = train_series.tail(self.config.optimization_train_window)

        if self.config.optimization_val_window and len(val_series) > self.config.optimization_val_window:
            val_series = val_series.tail(self.config.optimization_val_window)
        
        # If validation is empty, use a small portion of training data for validation
        if len(val_series) == 0 and len(train_series) > 1:
            split_idx = max(1, len(train_series) // 2)
            val_series = train_series.iloc[split_idx:]
            train_series = train_series.iloc[:split_idx]

        return train_series, val_series

    def optimize_parameters(
        self,
        p_range: Optional[Tuple[int, int]] = None,
        d_range: Optional[Tuple[int, int]] = None,
        q_range: Optional[Tuple[int, int]] = None,
    ) -> ParameterSearchResult:
        """
        Optimize ARIMA parameters using walk-forward validation.
        
        Args:
            p_range: Override p range from config
            d_range: Override d range from config
            q_range: Override q range from config
            
        Returns:
            ParameterSearchResult with best parameters and model
        """
        # Support legacy API with parameter overrides
        if p_range is not None or d_range is not None or q_range is not None:
            self.config = ARIMAConfig(
                target_column=self.config.target_column,
                p_range=p_range or self.config.p_range,
                d_range=d_range or self.config.d_range,
                q_range=q_range or self.config.q_range,
                early_stop_tolerance=self.config.early_stop_tolerance,
                optimization_train_window=self.config.optimization_train_window,
                optimization_val_window=self.config.optimization_val_window,
            )
        
        train_opt, val_opt = self._prepare_optimization_series()
        validator = ARIMAValidator(train_opt, val_opt)
        optimizer = ARIMAOptimizer(validator, self.config)
        result = optimizer.run()
        self.optimization_result = result
        self.best_params = result.best_params
        self.best_loss = result.best_loss
        self.trained_model = result.best_model
        self.best_model = result.best_model  # Backward compatibility alias
        return result

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------
    def train_final_model(self) -> ARIMA:
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize_parameters first.")

        combined = pd.concat([self.train_series, self.val_series]) if len(self.val_series) > 0 else self.train_series
        print(f"\nTraining final model with ARIMA{self.best_params} on combined train+val data...")
        self.trained_model = ARIMA(combined, order=self.best_params).fit()
        print(f"Final model AIC: {self.trained_model.aic:.2f}, BIC: {self.trained_model.bic:.2f}")
        return self.trained_model

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------
    def evaluate_on_test(self, test_window: Optional[int] = 500) -> ForecastResult:
        if self.best_params is None:
            raise ValueError("No trained model available. Run optimize_parameters and train_final_model first.")

        if test_window is not None and test_window < len(self.test_series):
            test_slice = self.test_series.tail(test_window)
            print(f"Evaluating on last {test_window} test points (out of {len(self.test_series)} total).")
        else:
            test_slice = self.test_series
            print(f"Evaluating on all {len(self.test_series)} test points. This may take a long time.")

        evaluator = ARIMAEvaluator(self.train_series, self.val_series, test_slice, self.best_params)
        forecast = evaluator.evaluate()
        self.last_forecast = forecast
        return forecast

    # --------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------
    def get_training_statistics(self) -> Dict[str, object]:
        if self.optimization_result is None:
            return {"message": "No training history available."}

        stats: Dict[str, object] = {
            'best_params': self.optimization_result.best_params,
            'best_val_loss': self.optimization_result.best_loss,
            'total_steps': len(self.optimization_result.steps),
            'early_stopped': self.optimization_result.early_stopped,
            'history': [step.__dict__ for step in self.optimization_result.steps],
        }

        if self.trained_model is not None:
            stats['final_aic'] = self.trained_model.aic
            stats['final_bic'] = self.trained_model.bic

        if self.last_forecast is not None:
            stats['test_mse'] = self.last_forecast.mse
            stats['test_rmse'] = self.last_forecast.rmse
            stats['test_mae'] = self.last_forecast.mae

        return stats

    def plot_results(self) -> None:
        if not self.ENABLE_PLOTTING:
            print("Plotting disabled. Set ARIMATrainer.ENABLE_PLOTTING = True to enable plots.")
            return

        if self.last_forecast is None:
            print("No cached forecast found. Evaluating on test data before plotting...")
            forecast = self.evaluate_on_test()
        else:
            forecast = self.last_forecast

        plotter = ARIMAPlotter(self.config.plot_max_points)
        plotter.plot(forecast, self.config.target_column)


def main() -> None:
    """
    Main training pipeline for ARIMA time series forecasting.
    
    Workflow:
    1. Load and split data via DataLoader
    2. Optimize ARIMA parameters using walk-forward validation
    3. Train final model on combined train+validation data
    4. Evaluate on hold-out test set with rolling forecasts
    5. Report statistics and visualize results
    """
    print("Loading data...")
    data_loader = DataLoader()

    config = ARIMAConfig(
        target_column='close',
        p_range=(0, 2),
        d_range=(0, 1),
        q_range=(0, 2),
        optimization_train_window=1000,
        optimization_val_window=100,
    )

    trainer = ARIMATrainer(data_loader, config)

    print(f"Train data shape: {trainer.train_series.shape}")
    print(f"Validation data shape: {trainer.val_series.shape}")
    print(f"Test data shape: {trainer.test_series.shape}")

    optimization_result = trainer.optimize_parameters()
    trainer.train_final_model()
    forecast = trainer.evaluate_on_test()

    stats = trainer.get_training_statistics()
    print("\nTraining Statistics:")
    for key, value in stats.items():
        if key != 'history':
            print(f"{key}: {value}")

    trainer.plot_results()


if __name__ == "__main__":
    main()
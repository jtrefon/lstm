from config.config_loader import LSTMConfig
from domain.models import ParameterSet
from .lstm_model import LSTMModel
import torch

def build_lstm(config: LSTMConfig, params: ParameterSet, device: torch.device) -> LSTMModel:
    if params.layers <= 1 and params.dropout > 0.0:
        raise ValueError("Invalid configuration: dropout > 0 requires layers > 1 for PyTorch LSTM.")
    return LSTMModel(
        input_size=int(config.model.input_size),
        hidden_size=int(params.units),
        num_layers=int(params.layers),
        dropout=float(params.dropout),
    ).to(device)

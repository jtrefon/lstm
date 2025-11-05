import logging
import torch

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Select the best available device and apply fast defaults.

    Note: Deterministic mode, when enabled elsewhere, should override cuDNN benchmark.
    """
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        dev = torch.device('cuda')
    elif torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    logger.info(f"Using device: {dev}")
    return dev

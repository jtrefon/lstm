import os
import pickle
import json
import time
import hashlib
from typing import Any, Dict, Optional, Tuple

import torch

from domain.models import ParameterSet
from domain.ports import ModelRepository


class TorchModelRepository(ModelRepository):
    def save_model(self, state_dict: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(state_dict, path)

    def save_scaler(self, scaler: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)


class ModelPackageRepository:
    def _hash_params(self, params: ParameterSet) -> str:
        raw = json.dumps(params.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _signature(self, params: ParameterSet, val_loss: Optional[float]) -> str:
        h = self._hash_params(params)
        ts = int(time.time())
        loss = f"{val_loss:.6f}" if val_loss is not None else "na"
        return f"{ts}_{h}_{loss}"

    def save_package(
        self,
        base_dir: str,
        state_dict: Dict[str, Any],
        scaler: Any,
        params: ParameterSet,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        sig = self._signature(params, (metrics or {}).get('val_loss'))
        out_dir = os.path.join(base_dir, sig)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(out_dir, 'model.pt'))
        with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(out_dir, 'params.json'), 'w') as f:
            json.dump(params.to_dict(), f, indent=2)
        meta = {'created_at': time.time()}
        if metrics:
            meta['metrics'] = metrics
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        return out_dir

    def load_package(
        self,
        dir_path: str,
    ) -> Tuple[Dict[str, Any], Any, ParameterSet, Dict[str, Any]]:
        state_dict = torch.load(os.path.join(dir_path, 'model.pt'), map_location='cpu')
        with open(os.path.join(dir_path, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(dir_path, 'params.json'), 'r') as f:
            params = ParameterSet.from_dict(json.load(f))
        metrics: Dict[str, Any] = {}
        meta_path = os.path.join(dir_path, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                m = json.load(f)
                metrics = m.get('metrics', {})
        return state_dict, scaler, params, metrics

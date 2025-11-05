import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import os
import sys

# Add src to path to resolve package imports like config.config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_results_path(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).resolve()
    cfg_path = ConfigLoader.get_config_path('optimization_config.yaml')
    ocfg = ConfigLoader.load_optimization_config(cfg_path)
    return (_project_root() / ocfg.persistence.results_file).resolve()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float('inf')


def _sort_by_loss(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: _to_float(r['val_loss']))


def _extract_params(r: Dict[str, str]) -> Dict[str, float]:
    keys = ['sequence_length', 'learning_rate', 'batch_size', 'units', 'layers', 'dropout']
    return {k: float(r[k]) for k in keys}


def _value_stats(rows: List[Dict[str, str]], key: str) -> Tuple[Counter, Dict[float, float]]:
    counts = Counter()
    loss_acc = defaultdict(float)
    loss_cnt = defaultdict(int)
    for r in rows:
        v = float(r[key])
        l = _to_float(r['val_loss'])
        counts[v] += 1
        loss_acc[v] += l
        loss_cnt[v] += 1
    means = {v: (loss_acc[v] / loss_cnt[v]) for v in loss_acc}
    return counts, means


def _top_values(counts: Counter, keep: int | None = None, frac: float | None = None) -> List[float]:
    items = counts.most_common()
    if frac is not None:
        total = sum(counts.values()) or 1
        acc = 0
        keep_vals = []
        for v, c in items:
            keep_vals.append(v)
            acc += c
            if acc / total >= frac:
                break
        return keep_vals
    if keep is not None:
        return [v for v, _ in items[:keep]]
    return [v for v, _ in items]


def _median_loss(rows: List[Dict[str, str]]) -> float:
    losses = sorted(_to_float(r['val_loss']) for r in rows)
    n = len(losses)
    if n == 0:
        return float('inf')
    mid = n // 2
    if n % 2:
        return losses[mid]
    return 0.5 * (losses[mid - 1] + losses[mid])


def _suggest_prune(all_rows: List[Dict[str, str]], top_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
    keys = ['sequence_length', 'learning_rate', 'batch_size', 'units', 'layers', 'dropout']
    med = _median_loss(all_rows)
    suggestions: Dict[str, Dict[str, List[float]]] = {}
    for k in keys:
        top_counts, _ = _value_stats(top_rows, k)
        all_counts, all_means = _value_stats(all_rows, k)
        keep_by_presence = set(top_counts.keys())
        keep_by_quality = {v for v, mean in all_means.items() if mean <= med}
        keep = sorted({*keep_by_presence, *keep_by_quality})
        drop = sorted([v for v in all_counts.keys() if v not in keep])
        suggestions[k] = {
            'keep': keep,
            'drop': drop,
        }
    return suggestions


def _print_distribution(title: str, counts: Counter, means: Dict[float, float]) -> None:
    logger.info(title)
    for v, c in counts.most_common():
        logger.info(f"  {v:g}: count={c} mean_loss={means.get(v, float('nan')):.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=str, default=None)
    p.add_argument('--top', type=int, default=50)
    p.add_argument('--top-coverage', type=float, default=0.8)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    )

    path = _resolve_results_path(args.results)
    if not path.exists():
        logger.error(f"Missing results file: {path}")
        return

    rows = _read_rows(path)
    if not rows:
        logger.warning('No rows found')
        return

    sorted_rows = _sort_by_loss(rows)
    top_rows = sorted_rows[: args.top]

    logger.info('Top trials:')
    for i, r in enumerate(top_rows, 1):
        ps = _extract_params(r)
        logger.info(
            f"  {i:02d} loss={_to_float(r['val_loss']):.6f} "
            f"seq_len={int(ps['sequence_length'])} lr={ps['learning_rate']:.6g} "
            f"batch={int(ps['batch_size'])} units={int(ps['units'])} "
            f"layers={int(ps['layers'])} dropout={ps['dropout']:.4g}"
        )

    logger.info('\nTop distribution and means:')
    for k in ['sequence_length', 'learning_rate', 'batch_size', 'units', 'layers', 'dropout']:
        counts, means = _value_stats(top_rows, k)
        _print_distribution(f'- {k}', counts, means)

    logger.info('\nGlobal means by value:')
    for k in ['sequence_length', 'learning_rate', 'batch_size', 'units', 'layers', 'dropout']:
        _, means = _value_stats(rows, k)
        top_counts, _ = _value_stats(top_rows, k)
        ranked = sorted(means.items(), key=lambda kv: kv[1])
        kept = _top_values(top_counts, frac=args.top_coverage)
        rec = [v for v, _ in ranked if v in kept]
        logger.info(f"- {k}: recommended={rec}")

    logger.info('\nPruning suggestions:')
    pr = _suggest_prune(rows, top_rows)
    for k, d in pr.items():
        logger.info(f"- {k}: keep={d['keep']} drop={d['drop']}")


if __name__ == '__main__':
    main()

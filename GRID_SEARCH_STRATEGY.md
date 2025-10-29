# Grid Search Strategy - 7 Critical Hyperparameters

## The Problem

You correctly identified that grid search was only tuning 3 hyperparameters (units, layers, dropout). I've now expanded it to tune **7 critical hyperparameters**, but this creates a massive search space.

## Current Grid Size

With the new configuration:
- **Sequence length**: 3 values (30, 90, 180)
- **Learning rate**: 4 values (0.0001, 0.0005, 0.001, 0.005)
- **Batch size**: 4 values (16, 32, 64, 128)
- **Epochs**: 3 values (30, 100, 200)
- **Units**: 4 values (128, 256, 384, 512)
- **Layers**: 3 values (1, 2, 3)
- **Dropout**: 3 values (0.0, 0.15, 0.3)

**Total: 5,184 combinations** ⚠️

At ~2 minutes per model = **~10,368 minutes (7 days) on GPU!**

## Solution: Tiered Search Strategy

### Option 1: Fast Search (Recommended for Initial Exploration)
**Time: ~4 hours | Combinations: 192**

Reduce to most critical parameters:
```python
seq_length_values = [60, 120]  # 2 values (reduced)
learning_rate_values = [0.0005, 0.001]  # 2 values (reduced)
batch_size_values = [32, 64]  # 2 values (reduced)
epochs_values = [50]  # 1 value (fixed)
units_values = [128, 256, 512]  # 3 values
layers_values = [1, 2]  # 2 values (reduced)
dropout_values = [0.1, 0.2]  # 2 values (reduced)

Total: 2 × 2 × 2 × 1 × 3 × 2 × 2 = 192 combinations
```

### Option 2: Balanced Search (Recommended for Production)
**Time: ~24 hours | Combinations: 576**

Medium-sized grid:
```python
seq_length_values = [30, 90, 180]  # 3 values
learning_rate_values = [0.0005, 0.001, 0.005]  # 3 values (reduced)
batch_size_values = [32, 64]  # 2 values (reduced)
epochs_values = [50, 100]  # 2 values (reduced)
units_values = [128, 256, 512]  # 3 values
layers_values = [1, 2, 3]  # 3 values
dropout_values = [0.1, 0.2]  # 2 values (reduced)

Total: 3 × 3 × 2 × 2 × 3 × 3 × 2 = 648 combinations
```

### Option 3: Exhaustive Search (For Final Tuning)
**Time: 7 days | Combinations: 5,184**

Current full grid (all 7 parameters fully tuned)

## Recommended Approach: Two-Phase Search

### Phase 1: Fast Exploration (4 hours)
Use Option 1 to find approximate best regions

### Phase 2: Fine-Tuning (24 hours)
Based on Phase 1 results, narrow ranges and use Option 2

Example:
- If Phase 1 finds best_lr = 0.0005, use [0.0001, 0.0005, 0.001] in Phase 2
- If Phase 1 finds best_units = 512, use [256, 512] in Phase 2
- If Phase 1 finds best_seq_len = 120, use [60, 120, 180] in Phase 2

## Why These 7 Parameters Matter

### 1. **Sequence Length** (30-180 days)
- **Impact**: HIGH - Controls temporal context
- **Too short**: Misses long-term patterns
- **Too long**: Includes irrelevant old data

### 2. **Learning Rate** (0.0001-0.005)
- **Impact**: CRITICAL - Controls convergence
- **Too high**: Unstable, diverges
- **Too low**: Slow, gets stuck

### 3. **Batch Size** (16-128)
- **Impact**: HIGH - Affects gradient quality
- **Smaller**: Better generalization, slower
- **Larger**: Faster, may overfit

### 4. **Epochs** (30-200)
- **Impact**: MEDIUM - Training duration
- **Too few**: Underfitting
- **Too many**: Overfitting (though dropout helps)

### 5. **Units** (128-512)
- **Impact**: CRITICAL - Model capacity
- **Too few**: Can't learn patterns
- **Too many**: Overfitting, slow

### 6. **Layers** (1-3)
- **Impact**: MEDIUM - Model depth
- **1 layer**: Simple patterns only
- **3 layers**: Complex patterns, overfitting risk

### 7. **Dropout** (0.0-0.3)
- **Impact**: MEDIUM - Regularization
- **0.0**: No regularization, overfitting
- **0.3**: Strong regularization, may underfit

## Implementation

The code has been updated to search all 7 parameters. To use different grid sizes, modify `_parameter_grid()` in `LSTMOptimizer`:

```python
# For fast search:
seq_length_values = [60, 120]
learning_rate_values = [0.0005, 0.001]
batch_size_values = [32, 64]
epochs_values = [50]
units_values = [128, 256, 512]
layers_values = [1, 2]
dropout_values = [0.1, 0.2]

# For balanced search:
seq_length_values = [30, 90, 180]
learning_rate_values = [0.0005, 0.001, 0.005]
batch_size_values = [32, 64]
epochs_values = [50, 100]
units_values = [128, 256, 512]
layers_values = [1, 2, 3]
dropout_values = [0.1, 0.2]
```

## Recommendation

**Start with Option 1 (Fast Search)** to:
1. Verify the code works end-to-end
2. Get a baseline model
3. Identify which parameters matter most
4. Narrow down ranges for Phase 2

Then use **Option 2 (Balanced Search)** for production-quality hyperparameters.

## Summary

✅ Grid search now tunes **7 critical hyperparameters** instead of just 3
✅ You can choose search size based on available time/compute
✅ Recommended: Two-phase approach (fast exploration → fine-tuning)
✅ Current code supports all three options - just modify `_parameter_grid()`

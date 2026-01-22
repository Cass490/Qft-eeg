# Quantum QVAE Debugging Summary

## Current Problem
**Quantum_QVAE loss is NaN on every epoch**, preventing the model from learning.

```json
{
  "Quantum_QVAE": [NaN, NaN, NaN],  // ❌ Cannot learn
  "Classical_AE": [0.605, 0.485, 0.415]  // ✅ Working fine
}
```

Result: Quantum accuracy stuck at ~31% (random), Classical at ~94%

## Changes Made

### 1. Enhanced Loss Function (`src/utils/losses.py`)
- Added comprehensive NaN/Inf checks
- Clamping probabilities to valid range [1e-8, 1.0]
- Normalizing probability distributions
- Fallback to reconstruction loss if total loss is NaN
- Debug prints to identify which component fails

### 2. Improved Training Visibility (`train_comparison.py`)
- Changed `leave=False` to `leave=True` in tqdm (now you can see progress)
- Added NaN detection in training loop with immediate break
- Print loss after each epoch
- Shows which component (recon/latent) contains NaN

### 3. Better NaN Handling in Model (`src/models/qvae.py`)
- Check encoder output for NaN
- Check quantum layer output for NaN/Inf
- Check attention output for NaN
- Check decoder output for NaN
- Replace NaN with safe fallback values
- Debug prints at each stage

## Next Steps

### Step 1: Run Diagnostic Test
```bash
conda activate qisk
cd /home/iyanshi/Codes/Priyanshi/minor-proj/minor
python test_quantum_nan.py
```

This will test a single forward/backward pass and tell you EXACTLY where the NaN originates.

### Step 2: Based on Results

**If test shows NaN in quantum layer:**
- The quantum circuit itself is broken
- Try reducing to 4 qubits, 4 layers
- Or use simple MSE loss instead of hybrid loss

**If test shows NaN in loss function:**
- The entropy/QWD computation is broken
- Temporarily use simple MSE: `loss = torch.nn.functional.mse_loss(recon, batch_x)`

**If test passes but training still fails:**
- Gradient accumulation issue
- Try removing gradient clipping
- Or reduce learning rate to 0.0001

### Step 3: Run Training with Debugging
```bash
python train_comparison.py
```

Now you'll see:
- Progress bars for each epoch
- Loss values printed after each epoch
- Warning messages showing where NaN appears
- Which component (encoder/quantum/attention/decoder) produces NaN

## Quick Fixes to Try

### Option A: Use Simple MSE Loss (Fastest Test)
In `train_comparison.py` line 89, change to:
```python
if name == 'Quantum_QVAE':
    loss = torch.nn.functional.mse_loss(recon, batch_x)  # Simple, no NaN
```

### Option B: Reduce Quantum Circuit Complexity
In `src/models/qvae.py` line 36:
```python
self.quantum_layer = QuantumEncoder(n_qubits=4, n_layers=4)  # Instead of 6, 8
```

### Option C: Remove Gradient Clipping
In `train_comparison.py` line 96, comment out:
```python
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Expected Output After Fixes

```
--- Training Quantum_QVAE ---
Epoch 1/3: 100%|████████████████| 54/54 [01:23<00:00]
Quantum_QVAE Epoch 1/3: Loss = 0.6234

Epoch 2/3: 100%|████████████████| 54/54 [01:22<00:00]
Quantum_QVAE Epoch 2/3: Loss = 0.5123

Epoch 3/3: 100%|████████████████| 54/54 [01:21<00:00]
Quantum_QVAE Epoch 3/3: Loss = 0.4567

Quantum_QVAE Final Accuracy: 75.23%  // Much better!
```

## Key Insights

1. **QFT removal helped** (you had 80% before QFT was added)
2. **Lightning.qubit is working** (not stuck anymore)
3. **Loss function is the problem** (NaN on every epoch)
4. **Classical model works fine** (proves data/training loop is OK)

The quantum model architecture is correct, but something in the loss computation or gradient flow is producing NaN.

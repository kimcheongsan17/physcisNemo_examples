# Darcy Custom Experiments

Use this folder for code that branches away from the official PhysicsNeMo Darcy FNO example.

Suggested order:

1. Copy `train_fno_darcy.py` from the official repo.
2. Rename it to `train_fno_darcy_custom.py`.
3. Keep the baseline runnable before adding changes.
4. Add one custom idea per commit or notebook section.

Good first customizations:

- custom loss: `MSE + gradient_penalty`
- extra metrics: relative L2, max error, spectrum error
- model size sweep: FNO modes and latent channels
- resolution generalization: train at 64 or 128, validate at another resolution

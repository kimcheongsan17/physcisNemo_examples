# physcisNemo_examples

Colab-first study repo for NVIDIA PhysicsNeMo examples and custom research experiments.

The first target is the official Darcy FNO example:

- NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo
- Darcy FNO example: https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/darcy_fno

## Open In Colab

Open the starter notebook with:

```text
https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/darcy_fno_colab_starter.ipynb
```

## Repo Layout

```text
notebooks/
  darcy_fno_colab_starter.ipynb

configs/
  darcy_fno_colab_smoke.yaml
  darcy_fno_colab_medium.yaml

experiments/
  darcy_custom/
```

## Study Flow

1. Run the official Darcy FNO baseline in Colab.
2. Save checkpoints and logs to Google Drive.
3. Change only config values first: resolution, batch size, FNO modes, latent channels.
4. Add one custom idea at a time: custom loss, extra input features, validation metrics, or model block changes.
5. Keep each experiment small enough to smoke test before running a longer training job.

## Recommended First Experiments

- Compare `fno_modes`: 8, 12, 16.
- Compare `latent_channels`: 16, 32, 64.
- Add `MSE + gradient_penalty` loss.
- Add relative L2 validation metric.
- Try train-low-resolution and infer-higher-resolution behavior.

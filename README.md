# PhysicsNeMo Darcy PINO

Korean, cell-by-cell study materials for NVIDIA PhysicsNeMo's official-resolution Darcy Physics-Informed Neural Operator (PINO) workflow.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/darcy_pino_physicsnemo_colab.ipynb)

Adaptive fixed-vs-spatial-weighted experiment:

[![Open Adaptive PINO In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/darcy_adaptive_pino_physicsnemo_colab.ipynb)

Solid mechanics MeshGraphNet-style adaptive residual experiment:

[![Open Solid Adaptive MGN In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/solid_adaptive_mgn_colab.ipynb)

## Main files

- `notebooks/darcy_pino_physicsnemo_colab.ipynb` — the recommended Colab/Jupyter notebook.
- `scripts/darcy_pino_physicsnemo.py` — the same 26 cells in `# %%` percent format for VS Code, Jupyter-aware editors, or sequential Python execution.
- `notebooks/darcy_adaptive_pino_physicsnemo_colab.ipynb` — a controlled comparison between uniform and residual-driven spatial physics weighting.
- `scripts/darcy_adaptive_pino_physicsnemo.py` — the adaptive notebook in `# %%` percent format.
- `notebooks/solid_adaptive_mgn_colab.ipynb` — a lightweight solid mechanics MeshGraphNet-style Colab inspired by PhysicsNeMo's `deforming_plate` example, comparing fixed and adaptive graph residual losses.
- `scripts/solid_adaptive_mgn.py` — the solid adaptive MeshGraphNet notebook in `# %%` percent format.

Both versions contain the same:

- PhysicsNeMo 2.1.1 `FNO` and `PhysicsInformer` workflow
- NVIDIA Darcy dataset download and official 241-to-240 boundary crop
- 240 x 240 permeability-to-pressure operator learning
- data MSE plus Darcy PDE residual loss
- API-origin, tensor-shape, finite-value, and device diagnostics
- 50-epoch T4 training, live visualization, validation, and checkpoint export

## Recommended usage

Open the notebook with the Colab badge and run from top to bottom on a T4 GPU runtime. The Python companion uses Colab-style `/content` paths and is primarily intended for cell-by-cell execution through its `# %%` markers.

Generated datasets and the approximately 27 MB model checkpoint are intentionally not committed. The notebook downloads/regenerates them and can save checkpoints to Google Drive when needed.

## Adaptive experiment

The adaptive notebook reuses the official 240 x 240 data, FNO, PhysicsInformer, scaling, boundary crop, optimizer, and global physics coefficient from the baseline. Only the spatial distribution of the PDE loss changes. Weights are detached from autograd and normalized to mean one over the interior, so the comparison does not silently increase the total physics-loss scale.

Its comparison cell separates the common unweighted Darcy residual metric from each model's actual training objective: the existing PINO uses `mean(abs(residual))`, while adaptive PINO uses `mean(weight * abs(residual))`. Every loss plot uses the same 50-epoch x-axis as the baseline, labels the precise loss quantity on the y-axis, and shows both model variants in the legend.

`FULL_BASELINE_COMPARISON=True` matches the existing GitHub PINO training schedule: batch size 1, all 102 training samples per epoch, 50 epochs, and full validation after every epoch. Fixed and adaptive curves are epoch averages, not unrelated single-batch values. Use multiple random seeds before drawing performance conclusions. The permeability-gradient prior is disabled by default because strong-form residuals around discontinuous coefficients need separate numerical validation.

## Solid adaptive MeshGraphNet-style experiment

The solid notebook follows the idea notes around PhysicsNeMo's structural mechanics `deforming_plate` MeshGraphNet example. It does not attempt to reproduce the full DeepMind deforming-plate dataset run in Colab. Instead, it builds a small synthetic plate graph, a MeshGraphNet-style encoder/processor/decoder, and a graph solid-residual proxy so the fixed-vs-adaptive loss idea can compile and run quickly on a Colab GPU.

This is not a conflict with MeshGraphNet: MGN is the mesh message-passing backbone, while the adaptive method changes how the residual loss is spatially weighted. The right interpretation is that adaptive weighting redistributes training pressure toward high-residual/interface nodes; it is not a blanket guarantee that every global residual metric improves.

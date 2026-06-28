# PhysicsNeMo Darcy PINO

Korean, cell-by-cell study materials for NVIDIA PhysicsNeMo's official-resolution Darcy Physics-Informed Neural Operator (PINO) workflow.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/darcy_pino_physicsnemo_colab.ipynb)

## Main files

- `notebooks/darcy_pino_physicsnemo_colab.ipynb` — the recommended Colab/Jupyter notebook.
- `scripts/darcy_pino_physicsnemo.py` — the same 26 cells in `# %%` percent format for VS Code, Jupyter-aware editors, or sequential Python execution.

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

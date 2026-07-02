# PhysicsNeMo Darcy PINO

Korean, cell-by-cell study materials for NVIDIA PhysicsNeMo's official-resolution Darcy Physics-Informed Neural Operator (PINO) workflow.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/darcy_pino_physicsnemo_colab.ipynb)

Adaptive fixed-vs-spatial-weighted experiment:

[![Open Adaptive PINO In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/darcy_adaptive_pino_physicsnemo_colab.ipynb)

Solid mechanics basic MeshGraphNet-style smoke example:

[![Open Solid Basic MGN In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/solid_basic_mgn_colab.ipynb)

Official-style PhysicsNeMo deforming plate MeshGraphNet smoke example:

[![Open Official-Style Deforming Plate MGN In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/solid_deforming_plate_mgn_physicsnemo_colab.ipynb)

Solid mechanics MeshGraphNet-style adaptive residual follow-up experiment:

[![Open Solid Adaptive MGN In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimcheongsan17/physcisNemo_examples/blob/main/notebooks/solid_adaptive_mgn_colab.ipynb)

## Main files

- `notebooks/darcy_pino_physicsnemo_colab.ipynb` — the recommended Colab/Jupyter notebook.
- `scripts/darcy_pino_physicsnemo.py` — the same 26 cells in `# %%` percent format for VS Code, Jupyter-aware editors, or sequential Python execution.
- `notebooks/darcy_adaptive_pino_physicsnemo_colab.ipynb` — a controlled comparison between uniform and residual-driven spatial physics weighting.
- `scripts/darcy_adaptive_pino_physicsnemo.py` — the adaptive notebook in `# %%` percent format.
- `notebooks/solid_basic_mgn_colab.ipynb` — the baseline solid mechanics MeshGraphNet-style Colab inspired by PhysicsNeMo's `deforming_plate` example, using supervised displacement loss plus a uniform graph residual loss.
- `scripts/solid_basic_mgn.py` — the solid basic MeshGraphNet notebook in `# %%` percent format.
- `notebooks/solid_deforming_plate_mgn_physicsnemo_colab.ipynb` — an official-style Colab based on NVIDIA PhysicsNeMo's `examples/structural_mechanics/deforming_plate`, matching the official node/edge/output tensor contract in a lightweight smoke run.
- `scripts/solid_deforming_plate_mgn_physicsnemo.py` — the official-style deforming plate notebook in `# %%` percent format.
- `notebooks/solid_adaptive_mgn_colab.ipynb` — a follow-up solid mechanics MeshGraphNet-style experiment comparing fixed and adaptive graph residual losses.
- `scripts/solid_adaptive_mgn.py` — the solid adaptive follow-up notebook in `# %%` percent format.

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

## Solid basic MeshGraphNet-style example

The basic solid notebook is the first step for the structural mechanics side. It builds a small synthetic plate graph, a MeshGraphNet-style encoder/processor/decoder, and a uniform graph solid-residual proxy. The point is to check that the solid MGN baseline compiles and actually trains in Colab before adding adaptive weighting.

The notebook keeps checkpoint writing disabled by default (`SAVE_CHECKPOINT=False`), so running it in Colab does not save a model unless you explicitly opt in.

## Official-style PhysicsNeMo deforming plate MeshGraphNet example

The official-style deforming plate notebook is based on NVIDIA PhysicsNeMo's `examples/structural_mechanics/deforming_plate` recipe. The full official example uses the DeepMind deforming-plate dataset, tetrahedral meshes, autoregressive rollout, `HybridMeshGraphNet`, and the official tensor contract: node input dimension 3, edge feature dimension 8, and output dimension 4 for velocity xyz plus stress.

The Colab notebook keeps that tensor contract but replaces the large TFRecord dataset and full trainer with a small synthetic 3D plate graph so it can compile and train quickly on a T4 runtime. It also documents the full official pipeline commands for later reproduction outside the smoke run.

## Solid adaptive MeshGraphNet-style follow-up experiment

The adaptive solid notebook follows the idea notes around PhysicsNeMo's structural mechanics `deforming_plate` MeshGraphNet example. It does not attempt to reproduce the full DeepMind deforming-plate dataset run in Colab. Instead, it builds a small synthetic plate graph, a MeshGraphNet-style encoder/processor/decoder, and a graph solid-residual proxy so the fixed-vs-adaptive loss idea can compile and run quickly on a Colab GPU.

This is not a conflict with MeshGraphNet: MGN is the mesh message-passing backbone, while the adaptive method changes how the residual loss is spatially weighted. The right interpretation is that adaptive weighting redistributes training pressure toward high-residual/interface nodes; it is not a blanket guarantee that every global residual metric improves.

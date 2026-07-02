#!/usr/bin/env python3
"""Build a Colab-ready notebook based on PhysicsNeMo's official deforming_plate example."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "solid_deforming_plate_mgn_physicsnemo_colab.ipynb"
SCRIPT = ROOT / "scripts" / "solid_deforming_plate_mgn_physicsnemo.py"


def _lines(text: str) -> list[str]:
    return (textwrap.dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(text)}


def export_percent_script(cells: list[dict]) -> None:
    chunks = [
        "# -*- coding: utf-8 -*-\n",
        '"""Companion script for the official-style deforming plate MGN Colab notebook."""\n\n',
    ]
    for cell in cells:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            chunks.append("# %% [markdown]\n")
            for line in source.splitlines(keepends=True):
                chunks.append(f"# {line}" if line.strip() else "#\n")
        else:
            chunks.extend(["# %%\n", source])
        chunks.append("\n")
    SCRIPT.write_text("".join(chunks).rstrip() + "\n", encoding="utf-8")


cells = [
    markdown(
        r"""
        # Official-style PhysicsNeMo deforming plate MGN Colab

        이 노트북은 NVIDIA PhysicsNeMo 공식 예제
        `examples/structural_mechanics/deforming_plate`를 기준으로 만든 Colab smoke version입니다.

        중요한 정리:

        - 공식 예제는 **DeepMind MeshGraphNets deforming plate** 데이터셋 재구현입니다.
        - 모델은 PhysicsNeMo `HybridMeshGraphNet` 계열입니다.
        - full run은 평균 1271 nodes, 400 time steps, 1000 train samples 규모라 Colab smoke로 바로 돌리기엔 무겁습니다.
        - 그래서 여기서는 공식 tensor contract를 맞춘 작은 3D plate graph를 생성해 one-step 학습이 되는지 확인합니다.

        공식 원본:

        - `NVIDIA/physicsnemo/examples/structural_mechanics/deforming_plate`
        - `deforming_plate_dataset.py`
        - `helpers.py::add_world_edges`
        - `train.py`
        - `conf/config.yaml`
        """
    ),
    markdown(
        r"""
        ## 1. 공식 예제와 이 Colab smoke의 대응

        | 항목 | 공식 PhysicsNeMo `deforming_plate` | 이 Colab smoke |
        |---|---|---|
        | 데이터 | DeepMind TFRecord, COMSOL, irregular tetra mesh | synthetic small 3D tetra-like plate |
        | node input | node type one-hot, dim 3 | 동일하게 dim 3 |
        | edge input | mesh/world edge feature, dim 8 | 동일하게 dim 8 |
        | model output | velocity xyz + stress, dim 4 | 동일하게 dim 4 |
        | 모델 | `HybridMeshGraphNet` | Tiny MGN-style model |
        | 학습 | autoregressive, large dataset | one-step smoke training |

        공식 config의 핵심값:

        ```yaml
        batch_size: 1
        num_training_samples: 1000
        num_training_time_steps: 200
        num_input_features: 3
        num_output_features: 4
        num_edge_features: 8
        lr: 0.0001
        ```

        즉 이 노트북은 “공식 full reproduction”이 아니라,
        공식 구조를 Colab에서 빠르게 확인하는 축소판입니다.
        """
    ),
    code(
        r"""
        import math
        import random
        import time
        from dataclasses import dataclass
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch import nn

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", DEVICE)
        print("torch :", torch.__version__)
        """
    ),
    markdown(
        r"""
        ## 2. Synthetic 3D deforming plate graph

        공식 데이터셋은 tetrahedral mesh의 `cells`에서 mesh edge를 만들고,
        현재 `world_pos`에서 가까운 node끼리 world edge를 추가합니다.

        여기서는 작은 rectangular 3D plate를 만들고:

        - 왼쪽 끝은 clamped node
        - 가운데 일부는 moving node
        - 오른쪽 일부는 object/load-like node
        - mesh edge feature 4개와 world edge feature 4개를 합쳐 edge dim 8을 만듭니다.
        """
    ),
    code(
        r"""
        @dataclass
        class PlateSample:
            node_type: torch.Tensor
            node_features: torch.Tensor
            edge_index: torch.Tensor
            edge_attr: torch.Tensor
            mesh_pos: torch.Tensor
            world_pos: torch.Tensor
            target: torch.Tensor
            cells: np.ndarray


        def build_template(nx=10, ny=6, nz=2):
            xs = torch.linspace(0.0, 1.0, nx)
            ys = torch.linspace(-0.28, 0.28, ny)
            zs = torch.linspace(-0.025, 0.025, nz)
            grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)

            def node_id(i, j, k):
                return (i * ny + j) * nz + k

            # A small tetra-like cell list for visualization/source-contract only.
            cells = []
            for i in range(nx - 1):
                for j in range(ny - 1):
                    a = node_id(i, j, 0)
                    b = node_id(i + 1, j, 0)
                    c = node_id(i, j + 1, 0)
                    d = node_id(i + 1, j + 1, 0)
                    e = node_id(i, j, 1)
                    f = node_id(i + 1, j, 1)
                    g = node_id(i, j + 1, 1)
                    h = node_id(i + 1, j + 1, 1)
                    cells.extend([(a, b, c, e), (b, d, c, h), (b, f, e, h), (c, e, g, h)])
            return grid.float(), np.asarray(cells, dtype=np.int64)


        mesh_pos_template, cells_template = build_template()
        N_NODES = mesh_pos_template.shape[0]
        print("template nodes:", N_NODES)
        print("template tetra-like cells:", cells_template.shape[0])
        """
    ),
    code(
        r"""
        def radius_edges(pos, radius, exclude_self=True):
            dist = torch.cdist(pos, pos)
            mask = dist <= radius
            if exclude_self:
                mask.fill_diagonal_(False)
            return torch.nonzero(mask, as_tuple=False).T.contiguous()


        def unique_edges(edge_index):
            if edge_index.numel() == 0:
                return edge_index
            key = edge_index[0] * N_NODES + edge_index[1]
            order = torch.argsort(key)
            edge_index = edge_index[:, order]
            key = key[order]
            keep = torch.ones_like(key, dtype=torch.bool)
            keep[1:] = key[1:] != key[:-1]
            return edge_index[:, keep]


        def make_node_type(mesh_pos):
            # Official dataset maps node_type values {0, 1, 3} to one-hot dim 3.
            x = mesh_pos[:, 0]
            node_type = torch.zeros(mesh_pos.shape[0], dtype=torch.long)  # moving
            node_type[x < 0.08] = 3  # clamped
            node_type[x > 0.90] = 1  # object/load-like boundary
            return node_type


        def node_type_one_hot(node_type):
            mapping = torch.full_like(node_type, -1)
            mapping[node_type == 0] = 0
            mapping[node_type == 1] = 1
            mapping[node_type == 3] = 2
            return F.one_hot(mapping, num_classes=3).float()


        def edge_features(src, dst, mesh_pos, world_pos):
            mesh_disp = mesh_pos[src] - mesh_pos[dst]
            mesh_norm = torch.linalg.vector_norm(mesh_disp, dim=1, keepdim=True)
            world_disp = world_pos[src] - world_pos[dst]
            world_norm = torch.linalg.vector_norm(world_disp, dim=1, keepdim=True)
            return torch.cat([mesh_disp, mesh_norm, world_disp, world_norm], dim=1)


        mesh_edge_index_template = unique_edges(radius_edges(mesh_pos_template, radius=0.145))
        print("template directed mesh edges:", mesh_edge_index_template.shape[1])
        """
    ),
    code(
        r"""
        def manufactured_plate_state(mesh_pos, load_value, phase):
            x, y, z = mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2]
            gate = x.square() * (3.0 - 2.0 * x)
            bend = torch.sin(math.pi * x) * torch.cos(2.0 * math.pi * y)
            twist = torch.sin(phase + 2.0 * math.pi * x)

            disp_x = 0.015 * load_value * gate * y
            disp_y = -0.010 * load_value * gate * x
            disp_z = 0.075 * load_value * gate * bend + 0.012 * twist * gate
            world_pos = mesh_pos + torch.stack([disp_x, disp_y, disp_z], dim=1)

            # One-step target velocity plus stress proxy.
            vel_x = 0.010 * load_value * gate * (1.0 - x)
            vel_y = -0.008 * load_value * gate * y
            vel_z = 0.070 * load_value * gate * torch.cos(math.pi * x) * torch.cos(2.0 * math.pi * y)
            stress = (0.35 * load_value * (torch.abs(torch.gradient(disp_z.reshape(10, 6, 2), dim=0)[0]).reshape(-1) + 0.05)).unsqueeze(1)
            target = torch.cat([torch.stack([vel_x, vel_y, vel_z], dim=1), stress], dim=1)
            return world_pos, target


        def make_sample(load_value, phase):
            mesh_pos = mesh_pos_template.clone()
            node_type = make_node_type(mesh_pos)
            node_features = node_type_one_hot(node_type)
            world_pos, target = manufactured_plate_state(mesh_pos, load_value, phase)

            # Mesh edges plus extra world edges, mimicking helpers.py::add_world_edges.
            mesh_edges = mesh_edge_index_template
            world_edges = radius_edges(world_pos, radius=0.122)
            mesh_pairs = set((int(a), int(b)) for a, b in mesh_edges.T.tolist())
            world_pairs = [(int(a), int(b)) for a, b in world_edges.T.tolist() if (int(a), int(b)) not in mesh_pairs]
            if world_pairs:
                world_edges = torch.tensor(world_pairs, dtype=torch.long).T
                edge_index = unique_edges(torch.cat([mesh_edges, world_edges], dim=1))
            else:
                edge_index = mesh_edges

            src, dst = edge_index
            edge_attr = edge_features(src, dst, mesh_pos, world_pos)
            clamped = node_type == 3
            target = torch.where(clamped[:, None], torch.zeros_like(target), target)
            return PlateSample(node_type, node_features, edge_index, edge_attr, mesh_pos, world_pos, target, cells_template)


        train_samples = [make_sample(float(v), phase=0.17 * i) for i, v in enumerate(torch.linspace(0.65, 1.35, 32))]
        valid_samples = [make_sample(float(v), phase=0.11 * i + 0.3) for i, v in enumerate(torch.linspace(0.75, 1.25, 8))]
        sample = train_samples[0]
        print("train samples:", len(train_samples))
        print("valid samples:", len(valid_samples))
        print("node input dim:", sample.node_features.shape[1])
        print("edge feature dim:", sample.edge_attr.shape[1])
        print("output dim:", sample.target.shape[1])
        print("edges in sample:", sample.edge_index.shape[1])
        """
    ),
    code(
        r"""
        fig = plt.figure(figsize=(7, 5), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        color = sample.target[:, 3]
        image = ax.scatter(sample.world_pos[:, 0], sample.world_pos[:, 1], sample.world_pos[:, 2], c=color, cmap="inferno", s=24)
        ax.set_title("Synthetic official-shape deforming plate sample\ncolor = stress proxy")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.colorbar(image, ax=ax, shrink=0.7)
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 3. Tiny official-shape MeshGraphNet

        공식 full model은 `HybridMeshGraphNet`입니다.
        Colab smoke에서는 설치 이슈를 줄이기 위해 같은 input/output contract를 갖는 작은 MGN-style network를 씁니다.
        """
    ),
    code(
        r"""
        class MLP(nn.Module):
            def __init__(self, in_dim, out_dim, hidden_dim=64, layers=2):
                super().__init__()
                modules = []
                last = in_dim
                for _ in range(layers - 1):
                    modules.extend([nn.Linear(last, hidden_dim), nn.SiLU()])
                    last = hidden_dim
                modules.append(nn.Linear(last, out_dim))
                self.net = nn.Sequential(*modules)

            def forward(self, x):
                return self.net(x)


        class TinyOfficialShapeMGN(nn.Module):
            def __init__(self, node_dim=3, edge_dim=8, out_dim=4, hidden_dim=80, message_steps=6):
                super().__init__()
                self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim)
                self.edge_encoder = MLP(edge_dim, hidden_dim, hidden_dim)
                self.edge_processors = nn.ModuleList([MLP(3 * hidden_dim, hidden_dim, hidden_dim) for _ in range(message_steps)])
                self.node_processors = nn.ModuleList([MLP(2 * hidden_dim, hidden_dim, hidden_dim) for _ in range(message_steps)])
                self.decoder = MLP(hidden_dim, out_dim, hidden_dim)

            def forward(self, node_features, edge_index, edge_attr):
                h = self.node_encoder(node_features)
                e = self.edge_encoder(edge_attr)
                send, recv = edge_index
                for edge_mlp, node_mlp in zip(self.edge_processors, self.node_processors):
                    msg = edge_mlp(torch.cat([h[send], h[recv], e], dim=-1))
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, recv, msg)
                    h = h + node_mlp(torch.cat([h, agg], dim=-1))
                    e = e + msg
                return self.decoder(h)


        model = TinyOfficialShapeMGN().to(DEVICE)
        with torch.no_grad():
            out = model(sample.node_features.to(DEVICE), sample.edge_index.to(DEVICE), sample.edge_attr.to(DEVICE))
        print("model output:", tuple(out.shape))
        """
    ),
    markdown(
        r"""
        ## 4. Training

        Loss를 공식 target 구조에 맞춰 분리해서 봅니다.

        - `velocity_mse`: output `0:3`
        - `stress_mse`: output `3`
        - `total`: `velocity_mse + 0.2 * stress_mse`
        """
    ),
    code(
        r"""
        EPOCHS = 70
        LR = 2.5e-3
        STRESS_WEIGHT = 0.20
        SAVE_CHECKPOINT = False


        def to_device(sample):
            return (
                sample.node_features.to(DEVICE),
                sample.edge_index.to(DEVICE),
                sample.edge_attr.to(DEVICE),
                sample.target.to(DEVICE),
                sample.node_type.to(DEVICE),
            )


        def compute_losses(model, sample):
            x, edge_index, edge_attr, target, node_type = to_device(sample)
            pred = model(x, edge_index, edge_attr)
            moving_mask = node_type == 0
            vel = F.mse_loss(pred[moving_mask, 0:3], target[moving_mask, 0:3])
            stress = F.mse_loss(pred[moving_mask, 3:4], target[moving_mask, 3:4])
            total = vel + STRESS_WEIGHT * stress
            return {"total": total, "velocity_mse": vel, "stress_mse": stress, "prediction": pred}


        def evaluate(model, samples):
            model.eval()
            sums = {"total": 0.0, "velocity_mse": 0.0, "stress_mse": 0.0}
            with torch.no_grad():
                for s in samples:
                    losses = compute_losses(model, s)
                    for key in sums:
                        sums[key] += float(losses[key].detach())
            return {key: value / len(samples) for key, value in sums.items()}
        """
    ),
    code(
        r"""
        model = TinyOfficialShapeMGN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        generator = torch.Generator().manual_seed(SEED)
        history = {key: [] for key in ["epoch", "train_velocity_mse", "train_stress_mse", "valid_velocity_mse", "valid_stress_mse", "valid_total"]}

        start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = torch.randperm(len(train_samples), generator=generator).tolist()
            train_vel = 0.0
            train_stress = 0.0
            for idx in order:
                optimizer.zero_grad(set_to_none=True)
                losses = compute_losses(model, train_samples[int(idx)])
                losses["total"].backward()
                optimizer.step()
                train_vel += float(losses["velocity_mse"].detach())
                train_stress += float(losses["stress_mse"].detach())

            valid = evaluate(model, valid_samples)
            history["epoch"].append(epoch)
            history["train_velocity_mse"].append(train_vel / len(train_samples))
            history["train_stress_mse"].append(train_stress / len(train_samples))
            history["valid_velocity_mse"].append(valid["velocity_mse"])
            history["valid_stress_mse"].append(valid["stress_mse"])
            history["valid_total"].append(valid["total"])

            if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
                print(
                    f"deforming plate epoch {epoch:03d}/{EPOCHS} | "
                    f"valid velocity={valid['velocity_mse']:.4e} | "
                    f"valid stress={valid['stress_mse']:.4e} | "
                    f"total={valid['total']:.4e}"
                )

        print("elapsed:", f"{time.perf_counter() - start:.1f}s")
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        specs = [
            ("train_velocity_mse", "Train velocity loss", "MSE velocity xyz"),
            ("valid_velocity_mse", "Validation velocity loss", "MSE velocity xyz"),
            ("valid_stress_mse", "Validation stress loss", "MSE stress proxy"),
        ]
        for ax, (key, title, ylabel) in zip(axes, specs):
            ax.semilogy(history["epoch"], history[key], marker="o", markersize=3, label="official-shape MGN smoke")
            ax.set_title(title)
            ax.set_xlabel(f"Epoch ({len(train_samples)} one-step samples per epoch)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 5. Prediction visualization

        색은 stress 또는 error입니다. 공식 inference는 mesh surface animation을 저장하지만,
        여기서는 Colab smoke라 3D scatter로 빠르게 확인합니다.
        """
    ),
    code(
        r"""
        @torch.no_grad()
        def predict_sample(model, sample):
            model.eval()
            x, edge_index, edge_attr, target, node_type = to_device(sample)
            pred = model(x, edge_index, edge_attr).detach().cpu()
            return pred


        valid_sample = valid_samples[len(valid_samples) // 2]
        pred = predict_sample(model, valid_sample)
        target = valid_sample.target
        error = torch.linalg.vector_norm(pred[:, 0:3] - target[:, 0:3], dim=1)

        fig = plt.figure(figsize=(15, 4), constrained_layout=True)
        panels = [
            (target[:, 3], "target stress proxy", "inferno"),
            (pred[:, 3], "predicted stress proxy", "inferno"),
            (error, "velocity error norm", "magma"),
        ]
        for i, (values, title, cmap) in enumerate(panels, 1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            sc = ax.scatter(valid_sample.world_pos[:, 0], valid_sample.world_pos[:, 1], valid_sample.world_pos[:, 2], c=values, cmap=cmap, s=24)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.colorbar(sc, ax=ax, shrink=0.65)
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 6. Full official pipeline, if you want to reproduce later

        Full reproduction은 이 smoke notebook보다 훨씬 무겁습니다.

        ```bash
        git clone https://github.com/NVIDIA/physicsnemo.git
        cd physicsnemo/examples/structural_mechanics/deforming_plate
        pip install -r requirements.txt
        cd raw_dataset
        sh download_dataset.sh deforming_plate
        cd ..
        python preprocessor.py
        python train.py
        python inference.py
        ```

        Colab에서 바로 full run을 하지 않는 이유:

        - DeepMind TFRecord dataset download/preprocess가 큼
        - 공식 config는 1000 train samples, 최대 200~400 time steps 계열
        - README 기준 full training은 multi-GPU/H100급 시간이 필요함

        그래서 GitHub example로 올릴 때는 이 smoke notebook을 기본으로 두고,
        full official pipeline 명령을 reference로 남기는 게 안전합니다.
        """
    ),
    code(
        r"""
        metrics = evaluate(model, valid_samples)
        print("final validation metrics")
        for key, value in metrics.items():
            print(f"{key:<18} {value:.4e}")
        print()
        print("initial -> final")
        print("valid velocity:", f"{history['valid_velocity_mse'][0]:.4e}", "->", f"{history['valid_velocity_mse'][-1]:.4e}")
        print("valid stress  :", f"{history['valid_stress_mse'][0]:.4e}", "->", f"{history['valid_stress_mse'][-1]:.4e}")

        if SAVE_CHECKPOINT:
            path = Path("/content/solid_deforming_plate_official_shape_mgn.pt")
            torch.save({"state_dict": model.state_dict(), "history": history, "metrics": metrics}, path)
            print("saved:", path)
        else:
            print("SAVE_CHECKPOINT=False, so no checkpoint was written.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "include_colab_link": True, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SCRIPT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
export_percent_script(cells)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
print(f"Wrote {SCRIPT}")

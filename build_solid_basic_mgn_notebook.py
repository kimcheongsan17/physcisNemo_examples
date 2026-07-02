#!/usr/bin/env python3
"""Build a Colab-ready basic solid mechanics MeshGraphNet notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "solid_basic_mgn_colab.ipynb"
SCRIPT = ROOT / "scripts" / "solid_basic_mgn.py"


def _lines(text: str) -> list[str]:
    return (textwrap.dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def export_percent_script(cells: list[dict]) -> None:
    chunks = [
        "# -*- coding: utf-8 -*-\n",
        '"""Companion script for the solid basic MeshGraphNet Colab notebook."""\n\n',
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
        # Solid mechanics basic MeshGraphNet Colab

        이 노트북은 메모의 PhysicsNeMo `deforming_plate` / MeshGraphNet 아이디어를
        Colab에서 바로 실행 가능한 **기본 solid mechanics MGN 예제**로 정리한 것입니다.

        이번 버전의 목표는 adaptive가 아닙니다.

        - 작은 2D plate mesh graph를 만든다.
        - MeshGraphNet-style encoder / processor / decoder를 구현한다.
        - displacement supervised loss와 uniform solid residual loss로 학습한다.
        - loss가 실제로 줄어드는지 Colab에서 확인한다.

        공식 맥락:

        - NVIDIA PhysicsNeMo `examples/structural_mechanics/deforming_plate`
        - DeepMind MeshGraphNets deforming plate dataset
        - 공식 예제는 irregular tetrahedral mesh, autoregressive rollout, 큰 MeshGraphNet을 사용합니다.
        - 여기서는 Colab smoke run에 맞춰 구조를 설명용 2D plate graph로 축소합니다.
        """
    ),
    markdown(
        r"""
        ## 1. 기본 MGN에서 무엇을 학습하나?

        ```text
        mesh nodes, edges, material/boundary/load features
                    │
                    ▼
        MeshGraphNet-style encoder
                    │
                    ▼
        [edge message MLP + node update MLP] × K
                    │
                    ▼
        predicted displacement û = (ûx, ûy)
                    │
          ┌─────────┴───────────────────┐
          ▼                             ▼
        data loss                 graph solid residual
        MSE(û, u)             internal_force(û) + external_force
          └─────────┬───────────────────┘
                    ▼
        total loss = data loss + λ · uniform physics residual
        ```

        여기서는 모든 자유 node의 residual을 같은 비중으로 평균냅니다.
        즉 adaptive weighting은 아직 없습니다. 먼저 기본 solid MGN이 학습되는지 확인하는 단계입니다.
        """
    ),
    code(
        r"""
        # 2. Imports and reproducibility
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
        ## 2. Synthetic deforming plate graph

        공식 `deforming_plate` 예제는 3D irregular mesh와 시간 rollout을 다룹니다.
        여기서는 빠른 Colab 확인을 위해 하나의 작은 2D plate graph를 만들고,
        load 크기만 바꾼 여러 sample을 생성합니다.

        각 node feature:

        | feature | 의미 |
        |---|---|
        | `x, y` | reference mesh position |
        | `material` | soft inclusion을 나타내는 stiffness proxy |
        | `is_clamp` | 왼쪽 고정 경계 |
        | `is_load` | 오른쪽 load 경계 |
        | `is_free` | residual을 평가할 자유 node |
        | `force_x, force_y` | manufactured equilibrium force |

        target은 displacement `u=(u_x,u_y)`입니다.
        """
    ),
    code(
        r"""
        @dataclass
        class PlateGraph:
            pos: torch.Tensor
            edge_index: torch.Tensor
            edge_attr: torch.Tensor
            edge_stiffness: torch.Tensor
            triangles: np.ndarray
            material: torch.Tensor
            clamp_mask: torch.Tensor
            load_mask: torch.Tensor
            free_mask: torch.Tensor
            interface_mask: torch.Tensor


        def build_plate_graph(nx=18, ny=12):
            xs = torch.linspace(0.0, 1.0, nx)
            ys = torch.linspace(0.0, 0.65, ny)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
            pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

            def node_id(i, j):
                return i * ny + j

            triangles = []
            edge_pairs = set()
            for i in range(nx - 1):
                for j in range(ny - 1):
                    a = node_id(i, j)
                    b = node_id(i + 1, j)
                    c = node_id(i, j + 1)
                    d = node_id(i + 1, j + 1)
                    triangles.extend([(a, b, c), (b, d, c)])
                    for u, v in [(a, b), (b, d), (d, c), (c, a), (a, d), (b, c)]:
                        edge_pairs.add((u, v))
                        edge_pairs.add((v, u))

            edge_index = torch.tensor(sorted(edge_pairs), dtype=torch.long).T
            send, recv = edge_index
            rel = pos[send] - pos[recv]
            dist = torch.linalg.vector_norm(rel, dim=1, keepdim=True).clamp_min(1e-6)
            edge_attr = torch.cat([rel, dist], dim=1)

            x, y = pos[:, 0], pos[:, 1]
            inclusion = ((x - 0.63) ** 2 / 0.075**2 + (y - 0.34) ** 2 / 0.10**2) < 1.0
            material = torch.where(inclusion, torch.full_like(x, 0.35), torch.ones_like(x))
            edge_stiffness = 0.5 * (material[send] + material[recv]).unsqueeze(1)

            clamp_mask = x < 1e-8
            load_mask = x > 1.0 - 1e-8
            free_mask = ~clamp_mask

            material_jump = (material[send] - material[recv]).abs()
            interface_score = torch.zeros_like(x)
            interface_score.index_add_(0, recv, material_jump)
            interface_mask = interface_score > 0

            return PlateGraph(
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_stiffness=edge_stiffness,
                triangles=np.asarray(triangles),
                material=material,
                clamp_mask=clamp_mask,
                load_mask=load_mask,
                free_mask=free_mask,
                interface_mask=interface_mask,
            )


        graph = build_plate_graph()
        print("nodes:", graph.pos.shape[0])
        print("directed edges:", graph.edge_index.shape[1])
        print("triangles:", graph.triangles.shape[0])
        print("interface nodes:", int(graph.interface_mask.sum()))
        """
    ),
    code(
        r"""
        def graph_internal_force(displacement, graph):
            send, recv = graph.edge_index
            rel_u = displacement[send] - displacement[recv]
            length2 = graph.edge_attr[:, 2:3].square().clamp_min(1e-6)
            spring_force = graph.edge_stiffness * rel_u / length2
            internal = torch.zeros_like(displacement)
            internal.index_add_(0, recv, spring_force)
            return internal


        def manufactured_displacement(pos, material, clamp_mask, load_value):
            x = pos[:, 0]
            y = pos[:, 1] / 0.65
            gate = x.square() * (3.0 - 2.0 * x)
            soft_amp = 1.0 + 0.40 * (1.0 - material)
            ux = 0.090 * load_value * gate * (0.55 + 0.45 * torch.sin(math.pi * y)) * soft_amp
            uy = -0.040 * load_value * gate * y * (1.0 - y) * (1.0 + 0.25 * torch.cos(math.pi * x))
            displacement = torch.stack([ux, uy], dim=1)
            return torch.where(clamp_mask[:, None], torch.zeros_like(displacement), displacement)


        def make_sample(load_value):
            target_u = manufactured_displacement(graph.pos, graph.material, graph.clamp_mask, load_value)
            force = -graph_internal_force(target_u, graph)
            force = torch.where(graph.clamp_mask[:, None], torch.zeros_like(force), force)
            node_features = torch.cat(
                [
                    graph.pos,
                    graph.material[:, None],
                    graph.clamp_mask[:, None].float(),
                    graph.load_mask[:, None].float(),
                    graph.free_mask[:, None].float(),
                    force,
                ],
                dim=1,
            )
            return {
                "node_features": node_features,
                "target_u": target_u,
                "force": force,
                "load": float(load_value),
            }


        train_loads = torch.linspace(0.65, 1.35, 24)
        valid_loads = torch.linspace(0.72, 1.28, 8)
        train_samples = [make_sample(float(v)) for v in train_loads]
        valid_samples = [make_sample(float(v)) for v in valid_loads]

        print("train samples:", len(train_samples))
        print("valid samples:", len(valid_samples))
        print("node feature dim:", train_samples[0]["node_features"].shape[1])
        print("target shape:", train_samples[0]["target_u"].shape)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        sample = train_samples[len(train_samples) // 2]
        for axis, values, title, cmap in [
            (axes[0], graph.material, "material / stiffness proxy", "viridis"),
            (axes[1], sample["target_u"][:, 0], "target ux", "coolwarm"),
            (axes[2], torch.linalg.vector_norm(sample["force"], dim=1), "manufactured force norm", "magma"),
        ]:
            image = axis.tripcolor(
                graph.pos[:, 0].numpy(),
                graph.pos[:, 1].numpy(),
                graph.triangles,
                values.detach().numpy(),
                shading="gouraud",
                cmap=cmap,
            )
            axis.triplot(graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy(), graph.triangles, lw=0.2, color="k", alpha=0.25)
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            plt.colorbar(image, ax=axis, fraction=0.046)
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 3. Tiny MeshGraphNet-style model

        PhysicsNeMo full MeshGraphNet은 더 큰 hidden size, 더 깊은 processor, 실제 dataset loader를 씁니다.
        여기서는 Colab 설명용으로 encoder / message passing processor / decoder 구조만 작게 구현합니다.

        이 기본 버전에서 확인할 것:

        - node feature와 edge feature가 MGN에 들어간다.
        - graph message passing 뒤에 displacement `û`를 예측한다.
        - supervised displacement loss와 uniform residual loss가 함께 줄어든다.
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


        class TinyMeshGraphNet(nn.Module):
            def __init__(self, node_dim, edge_dim=3, hidden_dim=64, message_steps=5):
                super().__init__()
                self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim)
                self.edge_encoder = MLP(edge_dim, hidden_dim, hidden_dim)
                self.edge_processors = nn.ModuleList(
                    [MLP(3 * hidden_dim, hidden_dim, hidden_dim) for _ in range(message_steps)]
                )
                self.node_processors = nn.ModuleList(
                    [MLP(2 * hidden_dim, hidden_dim, hidden_dim) for _ in range(message_steps)]
                )
                self.decoder = MLP(hidden_dim, 2, hidden_dim)

            def forward(self, node_features, edge_index, edge_attr):
                h = self.node_encoder(node_features)
                e = self.edge_encoder(edge_attr)
                send, recv = edge_index
                for edge_mlp, node_mlp in zip(self.edge_processors, self.node_processors):
                    message = edge_mlp(torch.cat([h[send], h[recv], e], dim=-1))
                    aggregate = torch.zeros_like(h)
                    aggregate.index_add_(0, recv, message)
                    h = h + node_mlp(torch.cat([h, aggregate], dim=-1))
                    e = e + message
                return self.decoder(h)


        NODE_DIM = train_samples[0]["node_features"].shape[1]
        test_model = TinyMeshGraphNet(NODE_DIM).to(DEVICE)
        with torch.no_grad():
            out = test_model(
                train_samples[0]["node_features"].to(DEVICE),
                graph.edge_index.to(DEVICE),
                graph.edge_attr.to(DEVICE),
            )
        print("model output:", tuple(out.shape))
        """
    ),
    markdown(
        r"""
        ## 4. Basic solid residual loss

        이 예제의 physics residual은 graph spring equilibrium proxy입니다.

        $$
        r_i = \left\|\sum_{j\in\mathcal{N}(i)}
        k_{ij}\frac{\hat u_j-\hat u_i}{\|x_j-x_i\|^2} + f_i\right\|_2
        $$

        기본 objective:

        $$
        L = \operatorname{MSE}(\hat u, u)
        + \lambda \left(
          \operatorname{mean}_{i\in free}(r_i)
          + \operatorname{mean}_{i\in clamp}\|\hat u_i\|_2
        \right)
        $$

        그래프를 그릴 때는 loss를 분리해서 봅니다.

        - `data MSE`: displacement supervised loss
        - `uniform residual`: 모든 자유 node 평균 solid residual
        - `interface residual`: soft inclusion 경계 node residual, 학습 objective는 아니고 관찰 metric
        """
    ),
    code(
        r"""
        PHYSICS_WEIGHT = 0.02
        EPOCHS = 60
        LR = 3e-3
        SAVE_CHECKPOINT = False


        def move_graph(graph, device):
            return PlateGraph(
                pos=graph.pos.to(device),
                edge_index=graph.edge_index.to(device),
                edge_attr=graph.edge_attr.to(device),
                edge_stiffness=graph.edge_stiffness.to(device),
                triangles=graph.triangles,
                material=graph.material.to(device),
                clamp_mask=graph.clamp_mask.to(device),
                load_mask=graph.load_mask.to(device),
                free_mask=graph.free_mask.to(device),
                interface_mask=graph.interface_mask.to(device),
            )


        graph_device = move_graph(graph, DEVICE)


        def compute_losses(model, sample):
            node_features = sample["node_features"].to(DEVICE)
            target_u = sample["target_u"].to(DEVICE)
            force = sample["force"].to(DEVICE)

            prediction = model(node_features, graph_device.edge_index, graph_device.edge_attr)
            internal = graph_internal_force(prediction, graph_device)
            residual_vec = internal + force
            residual_scalar = torch.linalg.vector_norm(residual_vec, dim=1)

            free = graph_device.free_mask
            clamp = graph_device.clamp_mask
            interface = graph_device.interface_mask & free

            data = F.mse_loss(prediction, target_u)
            uniform_residual = residual_scalar[free].mean()
            clamp_penalty = torch.linalg.vector_norm(prediction[clamp], dim=1).mean()
            interface_residual = residual_scalar[interface].mean()
            physics = uniform_residual + clamp_penalty
            total = data + PHYSICS_WEIGHT * physics

            return {
                "total": total,
                "data": data,
                "physics": physics,
                "uniform_residual": uniform_residual,
                "clamp_penalty": clamp_penalty,
                "interface_residual": interface_residual,
                "prediction": prediction,
                "residual_scalar": residual_scalar,
            }
        """
    ),
    code(
        r"""
        def evaluate(model, samples):
            model.eval()
            sums = {
                "total": 0.0,
                "data": 0.0,
                "physics": 0.0,
                "uniform_residual": 0.0,
                "clamp_penalty": 0.0,
                "interface_residual": 0.0,
            }
            with torch.no_grad():
                for sample in samples:
                    losses = compute_losses(model, sample)
                    for key in sums:
                        sums[key] += float(losses[key].detach())
            return {key: value / len(samples) for key, value in sums.items()}


        def train_basic_mgn():
            model = TinyMeshGraphNet(NODE_DIM).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            history = {key: [] for key in [
                "epoch",
                "train_total",
                "train_data",
                "train_uniform_residual",
                "train_interface_residual",
                "valid_total",
                "valid_data",
                "valid_uniform_residual",
                "valid_interface_residual",
            ]}
            generator = torch.Generator().manual_seed(SEED)
            start = time.perf_counter()
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = torch.randperm(len(train_samples), generator=generator).tolist()
                train_sums = {key: 0.0 for key in ["total", "data", "uniform_residual", "interface_residual"]}

                for sample_index in order:
                    sample = train_samples[int(sample_index)]
                    optimizer.zero_grad(set_to_none=True)
                    losses = compute_losses(model, sample)
                    losses["total"].backward()
                    optimizer.step()
                    for key in train_sums:
                        train_sums[key] += float(losses[key].detach())

                valid = evaluate(model, valid_samples)
                history["epoch"].append(epoch)
                history["train_total"].append(train_sums["total"] / len(train_samples))
                history["train_data"].append(train_sums["data"] / len(train_samples))
                history["train_uniform_residual"].append(train_sums["uniform_residual"] / len(train_samples))
                history["train_interface_residual"].append(train_sums["interface_residual"] / len(train_samples))
                history["valid_total"].append(valid["total"])
                history["valid_data"].append(valid["data"])
                history["valid_uniform_residual"].append(valid["uniform_residual"])
                history["valid_interface_residual"].append(valid["interface_residual"])

                if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
                    print(
                        f"basic epoch {epoch:03d}/{EPOCHS} | "
                        f"valid data={history['valid_data'][-1]:.4e} | "
                        f"valid residual={history['valid_uniform_residual'][-1]:.4e} | "
                        f"interface={history['valid_interface_residual'][-1]:.4e}"
                    )

            print("elapsed:", f"{time.perf_counter() - start:.1f}s")
            return model, history


        basic_model, basic_history = train_basic_mgn()
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        plot_specs = [
            ("train_data", "A. Train displacement data loss", "MSE(û, u)"),
            ("valid_data", "B. Validation displacement data loss", "MSE(û, u)"),
            ("valid_uniform_residual", "C. Validation uniform solid residual", "mean residual over free nodes"),
            ("valid_interface_residual", "D. Validation interface residual", "mean residual over interface nodes"),
        ]

        for axis, (key, title, ylabel) in zip(axes.flat, plot_specs):
            axis.semilogy(
                basic_history["epoch"],
                basic_history[key],
                marker="o",
                markersize=3,
                label="Basic MGN — uniform solid residual",
            )
            axis.set_title(title)
            axis.set_xlabel(f"Epoch ({len(train_samples)} optimizer updates per epoch)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=8)

        plt.show()
        """
    ),
    markdown(
        r"""
        ## 5. Deformed mesh visualization

        아래 그림은 validation sample 하나에서 target과 prediction을 비교합니다.
        색의 의미는 각 panel title과 colorbar에 적어두었습니다.
        """
    ),
    code(
        r"""
        @torch.no_grad()
        def collect_visuals(model, sample):
            model.eval()
            losses = compute_losses(model, sample)
            return {
                "target": sample["target_u"].detach().cpu(),
                "prediction": losses["prediction"].detach().cpu(),
                "residual": losses["residual_scalar"].detach().cpu(),
            }


        valid_sample = valid_samples[len(valid_samples) // 2]
        visuals = collect_visuals(basic_model, valid_sample)


        def plot_deformed(axis, displacement, values, title, cmap="viridis", scale=1.8):
            deformed = graph.pos + scale * displacement
            image = axis.tripcolor(
                deformed[:, 0].numpy(),
                deformed[:, 1].numpy(),
                graph.triangles,
                values.detach().numpy(),
                shading="gouraud",
                cmap=cmap,
            )
            axis.triplot(deformed[:, 0].numpy(), deformed[:, 1].numpy(), graph.triangles, lw=0.2, color="k", alpha=0.25)
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_xlabel("x + scaled ux")
            axis.set_ylabel("y + scaled uy")
            plt.colorbar(image, ax=axis, fraction=0.046)


        target_norm = torch.linalg.vector_norm(visuals["target"], dim=1)
        pred_norm = torch.linalg.vector_norm(visuals["prediction"], dim=1)
        error_norm = torch.linalg.vector_norm(visuals["prediction"] - visuals["target"], dim=1)

        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
        plot_deformed(axes[0], visuals["target"], target_norm, "Target deformation norm")
        plot_deformed(axes[1], visuals["prediction"], pred_norm, "Basic MGN prediction norm")
        plot_deformed(axes[2], visuals["prediction"], error_norm, "Prediction error norm", "magma")
        plot_deformed(axes[3], visuals["prediction"], visuals["residual"], "Solid residual norm", "inferno")
        plt.show()
        """
    ),
    code(
        r"""
        metrics = evaluate(basic_model, valid_samples)
        header = f"{'metric':<22} {'value':>12}"
        print(header)
        print("-" * len(header))
        for key in ["data", "uniform_residual", "interface_residual", "clamp_penalty", "total"]:
            print(f"{key:<22} {metrics[key]:>12.4e}")

        print()
        print("initial vs final validation:")
        print("valid data    :", f"{basic_history['valid_data'][0]:.4e}", "->", f"{basic_history['valid_data'][-1]:.4e}")
        print("valid residual:", f"{basic_history['valid_uniform_residual'][0]:.4e}", "->", f"{basic_history['valid_uniform_residual'][-1]:.4e}")
        """
    ),
    markdown(
        r"""
        ## 해석 메모

        이 노트북은 **solid 기본 MGN smoke example**입니다.
        따라서 여기서 말할 수 있는 claim은 작게 잡습니다.

        - MeshGraphNet-style message passing이 plate graph에서 displacement를 학습한다.
        - uniform solid residual을 loss에 넣으면 equilibrium proxy도 함께 관찰할 수 있다.
        - Colab T4에서 빠르게 컴파일/학습되는 기본 구조를 확인했다.

        아직 하지 않은 것:

        - 공식 DeepMind `deforming_plate` dataset 재현
        - autoregressive rollout evaluation
        - full PhysicsNeMo MeshGraphNet trainer 연결
        - adaptive residual weighting 비교

        다음 단계로 adaptive를 붙일 때는 이 basic notebook을 기준선으로 두고,
        loss weighting만 바꾸는 방식으로 비교하는 게 제일 깔끔합니다.
        """
    ),
    code(
        r"""
        if SAVE_CHECKPOINT:
            checkpoint_path = Path("/content/solid_basic_mgn_smoke.pt")
            torch.save(
                {
                    "source": "Basic Colab smoke example inspired by PhysicsNeMo deforming_plate MeshGraphNet",
                    "seed": SEED,
                    "epochs": EPOCHS,
                    "physics_weight": PHYSICS_WEIGHT,
                    "state_dict": basic_model.state_dict(),
                    "history": basic_history,
                    "metrics": metrics,
                },
                checkpoint_path,
            )
            print("saved:", checkpoint_path)
            print("size :", f"{checkpoint_path.stat().st_size / 1024**2:.2f} MB")
        else:
            print("SAVE_CHECKPOINT=False, so no checkpoint was written. Set it to True if you want a /content smoke checkpoint.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "include_colab_link": True,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
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

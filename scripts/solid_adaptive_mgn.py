# -*- coding: utf-8 -*-
"""Companion script for the solid adaptive MeshGraphNet Colab notebook."""

# %% [markdown]
# # Solid mechanics adaptive MeshGraphNet Colab
#
# 이 노트북은 메모의 PhysicsNeMo `deforming_plate` / MeshGraphNet 아이디어를
# Colab에서 바로 실행 가능한 작은 예제로 정리한 것입니다.
#
# 핵심 질문:
#
# > MeshGraphNet 자체와 충돌하지 않고, solid mechanics mesh 예제에서 residual이 큰 node/interface에 physics loss를 재배치할 수 있는가?
#
# 결론부터 말하면 **충돌하지 않습니다.** MeshGraphNet(MGN)은 unstructured mesh 위에서
# node/edge message passing을 하는 backbone이고, adaptive weighting은 그 backbone을 학습시킬 때
# loss의 공간 분포를 바꾸는 방식입니다.
#
# 이 노트북은 공식 대형 데이터셋을 그대로 학습하지 않습니다. 대신 Colab smoke run에 맞춰
# 작은 plate mesh와 manufactured displacement/force를 만들고,
# `fixed MGN`과 `adaptive MGN`을 같은 초기값/같은 데이터 순서로 비교합니다.
#
# 공식 맥락:
#
# - NVIDIA PhysicsNeMo `examples/structural_mechanics/deforming_plate`
# - DeepMind MeshGraphNets deforming plate dataset
# - PhysicsNeMo 예제는 평균 1271 nodes의 irregular tetrahedral mesh, 400 time steps, 1000 train samples를 사용합니다.
# - 여기서는 그 구조를 설명용 2D plate graph로 축소합니다.

# %% [markdown]
# ## 1. MGN과 adaptive loss는 어디가 다른가
#
# ```text
# mesh nodes, edges, material/boundary/load features
#             │
#             ▼
# MeshGraphNet-style encoder
#             │
#             ▼
# [edge message MLP + node update MLP] × K
#             │
#             ▼
# predicted displacement û = (ûx, ûy)
#             │
#   ┌─────────┴───────────────────┐
#   ▼                             ▼
# data loss                 graph solid residual
# MSE(û, u)             internal_force(û) + external_force
#   └─────────┬───────────────────┘
#             ▼
# total loss = data loss + λ · physics loss
# ```
#
# - **MGN**: mesh에서 정보를 전달하는 모델 구조입니다.
# - **fixed loss**: 모든 자유 node의 residual을 같은 비중으로 평균냅니다.
# - **adaptive loss**: residual이 큰 node에 더 큰 weight를 주되, 자유 node 평균 weight를 1로 정규화합니다.
#
# 따라서 adaptive 버전은 MGN을 대체하는 모델이 아니라,
# MGN 학습 objective를 조금 다르게 만든 실험입니다.

# %%
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

# %% [markdown]
# ## 2. Synthetic deforming plate graph
#
# 공식 `deforming_plate` 예제는 각 sample이 서로 다른 mesh를 가질 수 있는 구조입니다.
# 여기서는 Colab에서 빠르게 돌리기 위해 하나의 작은 rectangular plate graph를 만들고,
# load 크기만 바꾼 여러 sample을 생성합니다.
#
# 각 node feature:
#
# | feature | 의미 |
# |---|---|
# | `x, y` | reference mesh position |
# | `material` | soft inclusion을 나타내는 stiffness proxy |
# | `is_clamp` | 왼쪽 고정 경계 |
# | `is_load` | 오른쪽 load 경계 |
# | `is_free` | residual을 평가할 자유 node |
# | `force_x, force_y` | manufactured equilibrium force |
#
# target은 displacement `u=(u_x,u_y)`입니다.

# %%
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

# %%
def graph_internal_force(displacement, graph):
    send, recv = graph.edge_index
    rel_u = displacement[send] - displacement[recv]
    length2 = graph.edge_attr[:, 2:3].square().clamp_min(1e-6)
    spring_force = graph.edge_stiffness * rel_u / length2
    internal = torch.zeros_like(displacement)
    internal.index_add_(0, recv, spring_force)
    return internal


def manufactured_displacement(pos, material, load_value):
    x = pos[:, 0]
    y = pos[:, 1] / 0.65
    gate = x.square() * (3.0 - 2.0 * x)
    soft_amp = 1.0 + 0.40 * (1.0 - material)
    ux = 0.090 * load_value * gate * (0.55 + 0.45 * torch.sin(math.pi * y)) * soft_amp
    uy = -0.040 * load_value * gate * y * (1.0 - y) * (1.0 + 0.25 * torch.cos(math.pi * x))
    displacement = torch.stack([ux, uy], dim=1)
    displacement = torch.where(graph.clamp_mask[:, None], torch.zeros_like(displacement), displacement)
    return displacement


def make_sample(load_value):
    target_u = manufactured_displacement(graph.pos, graph.material, load_value)
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

print("node feature dim:", train_samples[0]["node_features"].shape[1])
print("target shape:", train_samples[0]["target_u"].shape)

# %%
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

# %% [markdown]
# ## 3. Tiny MeshGraphNet-style model
#
# PhysicsNeMo의 full MeshGraphNet은 훨씬 큰 hidden size와 processor depth를 사용합니다.
# 여기서는 Colab 설명용으로 encoder / message passing processor / decoder 구조만 작게 구현합니다.
#
# 이 구현의 목적:
#
# - node feature와 edge feature가 어떻게 MGN에 들어가는지 보여주기
# - `fixed` vs `adaptive` loss를 같은 backbone에서 비교하기
# - 컴파일과 smoke training이 빠르게 끝나게 하기

# %%
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

# %% [markdown]
# ## 4. Fixed vs adaptive solid residual
#
# 이 예제의 physics residual은 graph spring equilibrium proxy입니다.
#
# $$
# r_i = \left\|\sum_{j\in\mathcal{N}(i)}
# k_{ij}\frac{\hat u_j-\hat u_i}{\|x_j-x_i\|^2} + f_i\right\|_2
# $$
#
# fixed objective:
#
# $$
# L_{phys}^{fixed} = \operatorname{mean}_{i\in free}(r_i)
# $$
#
# adaptive objective:
#
# $$
# L_{phys}^{adaptive} = \operatorname{mean}_{i\in free}(w_i r_i),
# \qquad \operatorname{mean}_{i\in free}(w_i)=1
# $$
#
# `w_i`는 residual에서 만들지만 `detach()`해서, 모델이 weight 생성 경로로 우회하지 못하게 합니다.

# %%
PHYSICS_WEIGHT = 0.02
ADAPTIVE_ALPHA = 2.0
ADAPTIVE_TEMPERATURE = 0.40
EPOCHS = 60
LR = 3e-3


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


def adaptive_node_weights(residual_scalar, free_mask):
    scores = residual_scalar.detach()
    free_scores = scores[free_mask]
    median = torch.quantile(free_scores, 0.50)
    high = torch.quantile(free_scores, 0.90)
    normalized = (scores - median) / (high - median).abs().clamp_min(1e-6)
    weights = 1.0 + ADAPTIVE_ALPHA * torch.sigmoid(normalized / ADAPTIVE_TEMPERATURE)
    weights = torch.where(free_mask, weights, torch.ones_like(weights))
    weights = weights / weights[free_mask].mean().clamp_min(1e-6)
    return weights


def compute_losses(model, sample, mode):
    node_features = sample["node_features"].to(DEVICE)
    target_u = sample["target_u"].to(DEVICE)
    force = sample["force"].to(DEVICE)
    prediction = model(node_features, graph_device.edge_index, graph_device.edge_attr)
    internal = graph_internal_force(prediction, graph_device)
    residual_vec = internal + force
    residual_scalar = torch.linalg.vector_norm(residual_vec, dim=1)

    data = F.mse_loss(prediction, target_u)
    free = graph_device.free_mask
    clamp = graph_device.clamp_mask
    uniform_residual = residual_scalar[free].mean()
    clamp_penalty = torch.linalg.vector_norm(prediction[clamp], dim=1).mean()

    if mode == "adaptive":
        weights = adaptive_node_weights(residual_scalar, free)
    elif mode == "fixed":
        weights = torch.ones_like(residual_scalar)
    else:
        raise ValueError(f"unknown mode: {mode}")

    adaptive_residual = (weights[free] * residual_scalar[free]).mean()
    physics_objective = adaptive_residual + clamp_penalty
    total = data + PHYSICS_WEIGHT * physics_objective

    interface = graph_device.interface_mask & free
    interface_residual = residual_scalar[interface].mean()

    return {
        "total": total,
        "data": data,
        "physics_objective": physics_objective,
        "uniform_residual": uniform_residual,
        "interface_residual": interface_residual,
        "prediction": prediction,
        "residual_scalar": residual_scalar,
        "weights": weights,
    }

# %%
def evaluate(model, samples, mode):
    model.eval()
    sums = {
        "data": 0.0,
        "physics_objective": 0.0,
        "uniform_residual": 0.0,
        "interface_residual": 0.0,
        "total": 0.0,
    }
    with torch.no_grad():
        for sample in samples:
            losses = compute_losses(model, sample, mode)
            for key in sums:
                sums[key] += float(losses[key].detach())
    return {key: value / len(samples) for key, value in sums.items()}


def train_variant(mode, initial_state, epoch_orders):
    model = TinyMeshGraphNet(NODE_DIM).to(DEVICE)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {key: [] for key in [
        "epoch",
        "train_data",
        "train_uniform_residual",
        "train_interface_residual",
        "train_total",
        "valid_data",
        "valid_uniform_residual",
        "valid_interface_residual",
        "valid_total",
    ]}
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sums = {key: 0.0 for key in ["data", "uniform_residual", "interface_residual", "total"]}
        for sample_index in epoch_orders[epoch - 1]:
            sample = train_samples[int(sample_index)]
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, sample, mode)
            losses["total"].backward()
            optimizer.step()
            for key in train_sums:
                train_sums[key] += float(losses[key].detach())

        valid = evaluate(model, valid_samples, mode)
        history["epoch"].append(epoch)
        history["train_data"].append(train_sums["data"] / len(train_samples))
        history["train_uniform_residual"].append(train_sums["uniform_residual"] / len(train_samples))
        history["train_interface_residual"].append(train_sums["interface_residual"] / len(train_samples))
        history["train_total"].append(train_sums["total"] / len(train_samples))
        history["valid_data"].append(valid["data"])
        history["valid_uniform_residual"].append(valid["uniform_residual"])
        history["valid_interface_residual"].append(valid["interface_residual"])
        history["valid_total"].append(valid["total"])

        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            print(
                f"{mode:8s} epoch {epoch:03d}/{EPOCHS} | "
                f"valid data={history['valid_data'][-1]:.4e} | "
                f"valid residual={history['valid_uniform_residual'][-1]:.4e} | "
                f"interface={history['valid_interface_residual'][-1]:.4e}"
            )

    print(mode, "elapsed:", f"{time.perf_counter() - start:.1f}s")
    return model, history


base_model = TinyMeshGraphNet(NODE_DIM).to(DEVICE)
initial_state = {key: value.detach().clone() for key, value in base_model.state_dict().items()}
generator = torch.Generator().manual_seed(SEED)
epoch_orders = [torch.randperm(len(train_samples), generator=generator).tolist() for _ in range(EPOCHS)]

fixed_model, fixed_history = train_variant("fixed", initial_state, epoch_orders)
adaptive_model, adaptive_history = train_variant("adaptive", initial_state, epoch_orders)

# %%
labels = {
    "fixed": "Fixed MGN — uniform residual loss",
    "adaptive": "Adaptive MGN — residual-weighted loss",
}
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
plot_specs = [
    ("train_data", "A. Train displacement data loss", "MSE(û, u)"),
    ("train_uniform_residual", "B. Train common solid residual", "Uniform residual L1/L2 proxy"),
    ("train_interface_residual", "C. Train interface residual", "Interface residual proxy"),
    ("valid_data", "D. Validation displacement data loss", "MSE(û, u)"),
    ("valid_uniform_residual", "E. Validation common residual", "Uniform residual proxy"),
    ("valid_interface_residual", "F. Validation interface residual", "Interface residual proxy"),
]
for axis, (key, title, ylabel) in zip(axes.flat, plot_specs):
    axis.semilogy(fixed_history["epoch"], fixed_history[key], marker="o", markersize=3, label=labels["fixed"])
    axis.semilogy(adaptive_history["epoch"], adaptive_history[key], marker="s", markersize=3, label=labels["adaptive"])
    axis.set_title(title)
    axis.set_xlabel(f"Epoch ({len(train_samples)} optimizer updates per epoch)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## 5. Deformed mesh and adaptive weights
#
# 아래 그림은 같은 validation sample에서 target, fixed MGN, adaptive MGN을 비교합니다.
# 마지막 column은 adaptive run이 어느 node의 residual을 더 강하게 본 것인지 보여줍니다.

# %%
@torch.no_grad()
def collect_visuals(model, sample, mode):
    model.eval()
    losses = compute_losses(model, sample, mode)
    return {
        "target": sample["target_u"].detach().cpu(),
        "prediction": losses["prediction"].detach().cpu(),
        "residual": losses["residual_scalar"].detach().cpu(),
        "weights": losses["weights"].detach().cpu(),
    }


valid_sample = valid_samples[len(valid_samples) // 2]
fixed_vis = collect_visuals(fixed_model, valid_sample, "fixed")
adaptive_vis = collect_visuals(adaptive_model, valid_sample, "adaptive")

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


fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
target_norm = torch.linalg.vector_norm(fixed_vis["target"], dim=1)
fixed_error = torch.linalg.vector_norm(fixed_vis["prediction"] - fixed_vis["target"], dim=1)
adaptive_error = torch.linalg.vector_norm(adaptive_vis["prediction"] - adaptive_vis["target"], dim=1)

plot_deformed(axes[0, 0], fixed_vis["target"], target_norm, "Target deformation norm")
plot_deformed(axes[0, 1], fixed_vis["prediction"], fixed_error, "Fixed MGN error norm", "magma")
plot_deformed(axes[0, 2], adaptive_vis["prediction"], adaptive_error, "Adaptive MGN error norm", "magma")
plot_deformed(axes[0, 3], adaptive_vis["prediction"], adaptive_vis["weights"], "Adaptive residual weight", "plasma")

plot_deformed(axes[1, 0], fixed_vis["prediction"], fixed_vis["residual"], "Fixed residual", "inferno")
plot_deformed(axes[1, 1], adaptive_vis["prediction"], adaptive_vis["residual"], "Adaptive residual", "inferno")
plot_deformed(axes[1, 2], fixed_vis["prediction"], graph.interface_mask.float(), "Interface mask", "cool")
plot_deformed(axes[1, 3], adaptive_vis["prediction"], graph.material, "Material stiffness proxy", "viridis")
plt.show()

# %%
fixed_metrics = evaluate(fixed_model, valid_samples, "fixed")
adaptive_metrics = evaluate(adaptive_model, valid_samples, "adaptive")
header = f"{'model':<14} {'data MSE':>12} {'uniform residual':>18} {'interface residual':>20} {'total':>12}"
print(header)
print("-" * len(header))
for name, metrics in [("Fixed MGN", fixed_metrics), ("Adaptive MGN", adaptive_metrics)]:
    print(
        f"{name:<14} {metrics['data']:>12.4e} "
        f"{metrics['uniform_residual']:>18.4e} "
        f"{metrics['interface_residual']:>20.4e} "
        f"{metrics['total']:>12.4e}"
    )

# %% [markdown]
# ## 해석 메모
#
# 이 노트북은 PhysicsNeMo full `deforming_plate` 학습을 대체하지 않습니다.
# 공식 예제는 DeepMind 데이터셋, autoregressive rollout, 큰 MeshGraphNet, 긴 학습 시간이 필요합니다.
# 여기서는 메모의 아이디어를 Colab에서 확인 가능한 형태로 줄였습니다.
#
# 읽는 법:
#
# - MGN과 adaptive residual weighting은 충돌하지 않습니다.
# - MGN은 graph backbone이고, adaptive는 loss weighting입니다.
# - adaptive가 전체 residual 평균을 항상 낮춘다고 쓰면 과장입니다.
# - 좋은 claim은 "interface/큰 residual 영역으로 학습 압력을 재배치한다"입니다.
# - 실제 solid benchmark claim은 공식 `deforming_plate` 데이터로 seed/rollout/ablation까지 해야 합니다.
#
# 다음 확장:
#
# 1. PhysicsNeMo `deforming_plate_dataset.py` loader와 이 loss wrapper 연결
# 2. node type별 residual: clamp / free / load / contact 구분
# 3. rollout error와 one-step error 분리
# 4. stress 또는 strain proxy를 residual weight에 섞는 ablation
# 5. MGN vs AW-MGN vs Transformer/Transolver 계열 비교

# %%
checkpoint_path = Path("/content/solid_adaptive_mgn_smoke.pt")
torch.save(
    {
        "source": "Colab smoke example inspired by PhysicsNeMo deforming_plate MeshGraphNet",
        "seed": SEED,
        "epochs": EPOCHS,
        "physics_weight": PHYSICS_WEIGHT,
        "adaptive_alpha": ADAPTIVE_ALPHA,
        "fixed_state_dict": fixed_model.state_dict(),
        "adaptive_state_dict": adaptive_model.state_dict(),
        "fixed_history": fixed_history,
        "adaptive_history": adaptive_history,
        "fixed_metrics": fixed_metrics,
        "adaptive_metrics": adaptive_metrics,
    },
    checkpoint_path,
)
print("saved:", checkpoint_path)
print("size :", f"{checkpoint_path.stat().st_size / 1024**2:.2f} MB")

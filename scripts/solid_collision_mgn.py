# -*- coding: utf-8 -*-
"""Companion script for the solid collision MeshGraphNet Colab notebook."""

# %% [markdown]
# # Solid collision MeshGraphNet Colab
#
# 이 노트북은 **고체 충돌(contact/collision)** 을 위한 작은 Colab smoke example입니다.
# 앞의 `solid_basic_mgn_colab.ipynb`가 plate 변형 baseline이었다면, 이 노트북은 두 개의
# 2D elastic solid body가 서로 접근해 충돌하는 상황을 다룹니다.
#
# 목표:
#
# - 두 고체 body를 particle/mesh node 집합으로 표현합니다.
# - 내부 spring edge와 body 사이 contact edge를 만듭니다.
# - MeshGraphNet-style 모델이 충돌 후 node velocity를 예측하게 합니다.
# - data loss와 간단한 collision physics metric(momentum / separation)을 같이 봅니다.
#
# 아직 adaptive weighting은 넣지 않습니다. 먼저 **고체 충돌 기본 MGN이 학습되는지** 확인하고,
# 그 다음에 contact node 또는 penetration 영역에 adaptive weight를 주는 실험으로 확장하는 흐름입니다.

# %% [markdown]
# ## 1. 문제 구조
#
# ```text
# two solid bodies as node clouds
#      │
#      ├─ internal spring edges inside each solid
#      └─ dynamic contact edges between close nodes
#                  │
#                  ▼
# MeshGraphNet-style message passing
#                  │
#                  ▼
# predicted post-impact node velocity v̂(t + Δt)
#                  │
#      ┌───────────┴────────────────────┐
#      ▼                                ▼
# data loss                      collision metrics
# MSE(v̂, v*)          momentum error / separation velocity
# ```
#
# 이 예제의 target은 analytic rigid-body elastic collision rule에 작은 local deformation velocity를
# 섞어서 만든 synthetic data입니다. 따라서 정식 충돌 해석 benchmark가 아니라,
# Colab에서 빠르게 컴파일/학습/시각화 가능한 구조 확인용입니다.

# %%
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
# ## 2. Solid body template and collision samples
#
# 각 고체는 disk 안에 놓인 node cloud로 표현합니다.
#
# Node feature:
#
# | feature | 의미 |
# |---|---|
# | `x, y` | 현재 node 위치 |
# | `vx, vy` | 충돌 전 node velocity |
# | `body_A, body_B` | body id one-hot |
# | `rx, ry` | body 중심 기준 reference offset |
# | `is_contact_node` | 상대 body node와 가까운 contact 후보 |
#
# Edge feature:
#
# | feature | 의미 |
# |---|---|
# | `dx, dy, dist` | receiver 기준 상대 위치 |
# | `is_internal` | 같은 body 내부 spring edge |
# | `is_contact` | 다른 body와의 contact candidate edge |
# | `gap` | contact pair의 거리 여유 |

# %%
BODY_RADIUS = 0.22
PARTICLE_RADIUS = 0.045
CONTACT_CUTOFF = 2.35 * PARTICLE_RADIUS
INTERNAL_CUTOFF = 0.092
DT = 0.045
RESTITUTION = 0.90


def make_disk_offsets(radius=BODY_RADIUS, spacing=0.070):
    coords = []
    values = np.arange(-radius * 0.82, radius * 0.82 + 1e-9, spacing)
    for x in values:
        for y in values:
            if x * x + y * y <= (radius * 0.86) ** 2:
                coords.append((x, y))
    offsets = torch.tensor(coords, dtype=torch.float32)
    # Stable sort by x then y for reproducible edge order.
    order = torch.argsort(offsets[:, 0] * 1000.0 + offsets[:, 1])
    return offsets[order]


template_offsets = make_disk_offsets()
N_BODY = template_offsets.shape[0]
N_NODES = 2 * N_BODY
body_id = torch.cat([torch.zeros(N_BODY, dtype=torch.long), torch.ones(N_BODY, dtype=torch.long)])
ref_offsets = torch.cat([template_offsets, template_offsets], dim=0)
print("nodes per body:", N_BODY)
print("total nodes:", N_NODES)

# %%
def elastic_collision(v_a, v_b, center_a, center_b, restitution=RESTITUTION):
    normal = center_b - center_a
    normal = normal / torch.linalg.vector_norm(normal).clamp_min(1e-6)
    relative = v_a - v_b
    approach_speed = torch.dot(relative, normal)
    if approach_speed <= 0:
        return v_a, v_b, normal, approach_speed

    # Equal mass elastic collision impulse along the contact normal.
    impulse = 0.5 * (1.0 + restitution) * approach_speed
    v_a_after = v_a - impulse * normal
    v_b_after = v_b + impulse * normal
    return v_a_after, v_b_after, normal, approach_speed


def build_edges(pos):
    senders = []
    receivers = []
    attrs = []

    # Internal solid edges.
    for start in [0, N_BODY]:
        end = start + N_BODY
        for i in range(start, end):
            for j in range(start, end):
                if i == j:
                    continue
                rel = pos[i] - pos[j]
                dist = torch.linalg.vector_norm(rel)
                if dist <= INTERNAL_CUTOFF:
                    senders.append(i)
                    receivers.append(j)
                    attrs.append([rel[0], rel[1], dist, 1.0, 0.0, 0.0])

    # Dynamic contact candidate edges.
    for i in range(0, N_BODY):
        for j in range(N_BODY, N_NODES):
            rel = pos[i] - pos[j]
            dist = torch.linalg.vector_norm(rel)
            if dist <= CONTACT_CUTOFF:
                gap = dist - 2.0 * PARTICLE_RADIUS
                for u, v, r in [(i, j, rel), (j, i, -rel)]:
                    senders.append(u)
                    receivers.append(v)
                    attrs.append([r[0], r[1], dist, 0.0, 1.0, gap])

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def make_sample(seed):
    rng = np.random.default_rng(seed)
    y_offset = float(rng.uniform(-0.070, 0.070))
    x_gap = float(rng.uniform(0.006, 0.050))
    center_a = torch.tensor([-BODY_RADIUS - x_gap / 2.0, 0.50 * y_offset], dtype=torch.float32)
    center_b = torch.tensor([ BODY_RADIUS + x_gap / 2.0, -0.50 * y_offset], dtype=torch.float32)

    speed = float(rng.uniform(0.85, 1.25))
    tangent = float(rng.uniform(-0.18, 0.18))
    v_a = torch.tensor([ speed, tangent], dtype=torch.float32)
    v_b = torch.tensor([-speed * float(rng.uniform(0.88, 1.08)), -0.45 * tangent], dtype=torch.float32)

    v_a_after, v_b_after, normal, approach_speed = elastic_collision(v_a, v_b, center_a, center_b)

    pos_a = center_a + template_offsets
    pos_b = center_b + template_offsets
    pos = torch.cat([pos_a, pos_b], dim=0)
    pre_velocity = torch.cat([
        v_a.repeat(N_BODY, 1),
        v_b.repeat(N_BODY, 1),
    ], dim=0)

    # Local compressive deformation velocity near the contact side.
    local_a = torch.relu((template_offsets @ normal - 0.35 * BODY_RADIUS) / (0.55 * BODY_RADIUS))
    local_b = torch.relu((template_offsets @ (-normal) - 0.35 * BODY_RADIUS) / (0.55 * BODY_RADIUS))
    deform_scale = 0.10 * approach_speed.clamp_min(0.0)
    target_a = v_a_after.repeat(N_BODY, 1) - deform_scale * local_a[:, None] * normal
    target_b = v_b_after.repeat(N_BODY, 1) + deform_scale * local_b[:, None] * normal
    target_velocity = torch.cat([target_a, target_b], dim=0)

    edge_index, edge_attr = build_edges(pos)
    contact_node = torch.zeros(N_NODES, dtype=torch.float32)
    if edge_attr.shape[0] > 0:
        contact_edges = edge_attr[:, 4] > 0.5
        if contact_edges.any():
            contact_node.index_fill_(0, edge_index[0, contact_edges], 1.0)
            contact_node.index_fill_(0, edge_index[1, contact_edges], 1.0)

    body_one_hot = F.one_hot(body_id, num_classes=2).float()
    node_features = torch.cat(
        [
            pos,
            pre_velocity,
            body_one_hot,
            ref_offsets,
            contact_node[:, None],
        ],
        dim=1,
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "target_velocity": target_velocity,
        "pre_velocity": pre_velocity,
        "pos": pos,
        "body_id": body_id.clone(),
        "normal": normal,
        "approach_speed": approach_speed,
        "centers": torch.stack([center_a, center_b]),
        "post_body_velocity": torch.stack([v_a_after, v_b_after]),
    }


train_samples = [make_sample(SEED + i) for i in range(48)]
valid_samples = [make_sample(10_000 + i) for i in range(12)]

sample = train_samples[0]
print("train samples:", len(train_samples))
print("valid samples:", len(valid_samples))
print("node feature dim:", sample["node_features"].shape[1])
print("edge feature dim:", sample["edge_attr"].shape[1])
print("edges in sample:", sample["edge_index"].shape[1])
print("contact edges in sample:", int((sample["edge_attr"][:, 4] > 0.5).sum()))

# %%
def plot_collision_sample(sample, title="synthetic solid collision sample"):
    pos = sample["pos"]
    edge_index = sample["edge_index"]
    edge_attr = sample["edge_attr"]
    contact_edges = edge_attr[:, 4] > 0.5

    fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    colors = ["tab:blue" if b == 0 else "tab:orange" for b in sample["body_id"].tolist()]
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=36, zorder=3)
    # Draw a subset of contact edges.
    contact_ids = torch.where(contact_edges)[0][:80]
    for edge_id in contact_ids:
        u = int(edge_index[0, edge_id])
        v = int(edge_index[1, edge_id])
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], color="red", alpha=0.25, lw=1.0)
    velocity = sample["pre_velocity"]
    ax.quiver(pos[:, 0], pos[:, 1], velocity[:, 0], velocity[:, 1], angles="xy", scale_units="xy", scale=8.0, alpha=0.45)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    plt.show()


plot_collision_sample(train_samples[0])

# %% [markdown]
# ## 3. Tiny MeshGraphNet-style collision model
#
# 여기서는 PhysicsNeMo full trainer를 쓰지 않고, Colab에서 바로 보이는 작은 MGN-style model을 씁니다.
#
# - Encoder: node/edge feature를 hidden state로 올림
# - Processor: edge message + node update를 여러 번 반복
# - Decoder: 충돌 후 node velocity `v(t+Δt)` 예측

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


class TinyCollisionMeshGraphNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, message_steps=5):
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
EDGE_DIM = train_samples[0]["edge_attr"].shape[1]
smoke_model = TinyCollisionMeshGraphNet(NODE_DIM, EDGE_DIM).to(DEVICE)
with torch.no_grad():
    smoke_out = smoke_model(
        train_samples[0]["node_features"].to(DEVICE),
        train_samples[0]["edge_index"].to(DEVICE),
        train_samples[0]["edge_attr"].to(DEVICE),
    )
print("model output:", tuple(smoke_out.shape))

# %% [markdown]
# ## 4. Collision losses and metrics
#
# 학습 objective는 기본적으로 supervised velocity MSE입니다.
# 여기에 아주 작은 momentum consistency term을 더합니다.
#
# 관찰 metric:
#
# - `velocity_mse`: 전체 node의 post-impact velocity MSE
# - `contact_mse`: contact 후보 node에서의 velocity MSE
# - `momentum_error`: 두 body 평균 velocity의 총합 보존 오차
# - `separation_velocity`: 충돌 후 두 body가 contact normal 방향으로 멀어지는 정도
#
# 이 metric들은 고체 충돌 문제에서 나중에 adaptive weight를 줄 위치를 판단하는 기준이 됩니다.

# %%
EPOCHS = 70
LR = 2.5e-3
MOMENTUM_WEIGHT = 1.0e-3
SAVE_CHECKPOINT = False


def to_device_sample(sample):
    moved = {}
    for key, value in sample.items():
        moved[key] = value.to(DEVICE) if torch.is_tensor(value) else value
    return moved


def body_mean(values, body_id, body_index):
    return values[body_id == body_index].mean(dim=0)


def compute_losses(model, sample):
    sample = to_device_sample(sample)
    pred_velocity = model(sample["node_features"], sample["edge_index"], sample["edge_attr"])
    target = sample["target_velocity"]
    body = sample["body_id"]

    velocity_mse = F.mse_loss(pred_velocity, target)
    contact_mask = sample["node_features"][:, -1] > 0.5
    if contact_mask.any():
        contact_mse = F.mse_loss(pred_velocity[contact_mask], target[contact_mask])
    else:
        contact_mse = velocity_mse

    pred_a = body_mean(pred_velocity, body, 0)
    pred_b = body_mean(pred_velocity, body, 1)
    pre_a = body_mean(sample["pre_velocity"], body, 0)
    pre_b = body_mean(sample["pre_velocity"], body, 1)
    target_a = body_mean(target, body, 0)
    target_b = body_mean(target, body, 1)

    momentum_error = torch.linalg.vector_norm((pred_a + pred_b) - (pre_a + pre_b))
    target_momentum_error = torch.linalg.vector_norm((target_a + target_b) - (pre_a + pre_b))
    normal = sample["normal"]
    separation_velocity = torch.dot(pred_b - pred_a, normal)
    separation_penalty = torch.relu(0.0 - separation_velocity)

    total = velocity_mse + MOMENTUM_WEIGHT * (momentum_error + separation_penalty)
    return {
        "total": total,
        "velocity_mse": velocity_mse,
        "contact_mse": contact_mse,
        "momentum_error": momentum_error,
        "target_momentum_error": target_momentum_error,
        "separation_velocity": separation_velocity,
        "prediction": pred_velocity,
    }

# %%
def evaluate(model, samples):
    model.eval()
    sums = {
        "total": 0.0,
        "velocity_mse": 0.0,
        "contact_mse": 0.0,
        "momentum_error": 0.0,
        "target_momentum_error": 0.0,
        "separation_velocity": 0.0,
    }
    with torch.no_grad():
        for sample in samples:
            losses = compute_losses(model, sample)
            for key in sums:
                sums[key] += float(losses[key].detach())
    return {key: value / len(samples) for key, value in sums.items()}


def train_collision_mgn():
    model = TinyCollisionMeshGraphNet(NODE_DIM, EDGE_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {key: [] for key in [
        "epoch",
        "train_velocity_mse",
        "train_contact_mse",
        "valid_velocity_mse",
        "valid_contact_mse",
        "valid_momentum_error",
        "valid_separation_velocity",
    ]}
    generator = torch.Generator().manual_seed(SEED)
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_samples), generator=generator).tolist()
        train_sums = {"velocity_mse": 0.0, "contact_mse": 0.0}
        for sample_index in order:
            sample = train_samples[int(sample_index)]
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, sample)
            losses["total"].backward()
            optimizer.step()
            train_sums["velocity_mse"] += float(losses["velocity_mse"].detach())
            train_sums["contact_mse"] += float(losses["contact_mse"].detach())

        valid = evaluate(model, valid_samples)
        history["epoch"].append(epoch)
        history["train_velocity_mse"].append(train_sums["velocity_mse"] / len(train_samples))
        history["train_contact_mse"].append(train_sums["contact_mse"] / len(train_samples))
        history["valid_velocity_mse"].append(valid["velocity_mse"])
        history["valid_contact_mse"].append(valid["contact_mse"])
        history["valid_momentum_error"].append(valid["momentum_error"])
        history["valid_separation_velocity"].append(valid["separation_velocity"])

        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            print(
                f"collision epoch {epoch:03d}/{EPOCHS} | "
                f"valid velocity MSE={valid['velocity_mse']:.4e} | "
                f"contact MSE={valid['contact_mse']:.4e} | "
                f"momentum err={valid['momentum_error']:.4e} | "
                f"sep vel={valid['separation_velocity']:.4e}"
            )

    print("elapsed:", f"{time.perf_counter() - start:.1f}s")
    return model, history


collision_model, collision_history = train_collision_mgn()

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
plot_specs = [
    ("train_velocity_mse", "A. Train velocity loss", "MSE(v̂, v*)"),
    ("valid_velocity_mse", "B. Validation velocity loss", "MSE(v̂, v*)"),
    ("valid_contact_mse", "C. Validation contact-node loss", "contact MSE"),
    ("valid_momentum_error", "D. Validation momentum error", "||mean(vA)+mean(vB)-pre||"),
]
for axis, (key, title, ylabel) in zip(axes.flat, plot_specs):
    axis.semilogy(
        collision_history["epoch"],
        collision_history[key],
        marker="o",
        markersize=3,
        label="Collision MGN",
    )
    axis.set_title(title)
    axis.set_xlabel(f"Epoch ({len(train_samples)} collision samples per epoch)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## 5. Collision prediction visualization
#
# 아래 그림은 validation collision 하나에서
# 충돌 전 velocity, target post-impact velocity, MGN prediction, prediction error를 비교합니다.

# %%
@torch.no_grad()
def collect_prediction(model, sample):
    model.eval()
    losses = compute_losses(model, sample)
    return losses["prediction"].detach().cpu()


valid_sample = valid_samples[len(valid_samples) // 2]
pred_velocity = collect_prediction(collision_model, valid_sample)
target_velocity = valid_sample["target_velocity"]
pre_velocity = valid_sample["pre_velocity"]
pos = valid_sample["pos"]
body = valid_sample["body_id"]
colors = ["tab:blue" if b == 0 else "tab:orange" for b in body.tolist()]

fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
panels = [
    (pre_velocity, torch.linalg.vector_norm(pre_velocity, dim=1), "pre-impact velocity", "viridis"),
    (target_velocity, torch.linalg.vector_norm(target_velocity, dim=1), "target post-impact velocity", "viridis"),
    (pred_velocity, torch.linalg.vector_norm(pred_velocity, dim=1), "MGN predicted velocity", "viridis"),
    (pred_velocity - target_velocity, torch.linalg.vector_norm(pred_velocity - target_velocity, dim=1), "prediction error norm", "magma"),
]

for axis, (velocity, values, title, cmap) in zip(axes, panels):
    scatter = axis.scatter(pos[:, 0], pos[:, 1], c=values.numpy(), s=42, cmap=cmap, edgecolor="k", linewidth=0.2)
    axis.quiver(pos[:, 0], pos[:, 1], velocity[:, 0], velocity[:, 1], angles="xy", scale_units="xy", scale=8.0, alpha=0.65)
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.2)
    plt.colorbar(scatter, ax=axis, fraction=0.046)
plt.show()

# %%
metrics = evaluate(collision_model, valid_samples)
header = f"{'metric':<28} {'value':>12}"
print(header)
print("-" * len(header))
for key in ["velocity_mse", "contact_mse", "momentum_error", "target_momentum_error", "separation_velocity", "total"]:
    print(f"{key:<28} {metrics[key]:>12.4e}")

print()
print("initial vs final validation:")
print("valid velocity MSE:", f"{collision_history['valid_velocity_mse'][0]:.4e}", "->", f"{collision_history['valid_velocity_mse'][-1]:.4e}")
print("valid contact MSE :", f"{collision_history['valid_contact_mse'][0]:.4e}", "->", f"{collision_history['valid_contact_mse'][-1]:.4e}")

# %% [markdown]
# ## 해석 메모
#
# 이 노트북은 고체 충돌용 full benchmark가 아니라, collision/contact 구조를 Colab에서 빠르게 확인하는
# MeshGraphNet-style smoke example입니다.
#
# 여기서 말할 수 있는 것:
#
# - solid body를 node cloud와 internal/contact edge로 표현했다.
# - MGN이 충돌 후 velocity field를 학습한다.
# - contact node의 error와 momentum/separation metric을 따로 관찰할 수 있다.
#
# 다음 확장:
#
# 1. contact edge 또는 contact node에 adaptive residual weight 적용
# 2. penetration penalty를 loss에 직접 추가
# 3. velocity one-step이 아니라 multi-step rollout으로 충돌 후 분리 과정 평가
# 4. particle cloud 대신 tetra/tri mesh 기반 elastic solid로 확장

# %%
if SAVE_CHECKPOINT:
    checkpoint_path = Path("/content/solid_collision_mgn_smoke.pt")
    torch.save(
        {
            "source": "Solid collision MeshGraphNet-style Colab smoke example",
            "seed": SEED,
            "epochs": EPOCHS,
            "state_dict": collision_model.state_dict(),
            "history": collision_history,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print("saved:", checkpoint_path)
    print("size :", f"{checkpoint_path.stat().st_size / 1024**2:.2f} MB")
else:
    print("SAVE_CHECKPOINT=False, so no checkpoint was written. Set it to True if you want a /content smoke checkpoint.")

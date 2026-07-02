#!/usr/bin/env python3
"""Derive a Colab-ready adaptive PINO experiment from the official Darcy notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "notebooks" / "darcy_pino_physicsnemo_colab.ipynb"
OUTPUT = ROOT / "notebooks" / "darcy_adaptive_pino_physicsnemo_colab.ipynb"
SCRIPT = ROOT / "scripts" / "darcy_adaptive_pino_physicsnemo.py"


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
        '"""Companion script for the adaptive Darcy PINO Colab notebook."""\n\n',
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


baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

# Keep the baseline intact through its API/tensor diagnostics. Replace only the
# expensive 50-epoch training and final checkpoint cells with the controlled
# fixed-vs-adaptive experiment below.
cells = baseline["cells"][:22]
cells[0] = markdown(
    r"""
    # Adaptive Darcy PINO on PhysicsNeMo — fixed vs spatially weighted physics loss

    이 노트북은 기존 [`darcy_pino_physicsnemo_colab.ipynb`](darcy_pino_physicsnemo_colab.ipynb)의
    NVIDIA 공식 240×240 Darcy 설정을 그대로 준비한 뒤, **physics loss의 공간 가중치만** 바꾸어 비교합니다.

    - `fixed PINO`: 모든 내부 pixel의 PDE residual을 같은 비중으로 학습
    - `adaptive PINO`: 현재 residual이 큰 내부 pixel을 더 강하게 학습

    두 run은 같은 FNO 초기값, 같은 데이터 순서, 같은 optimizer와 같은 전역 physics coefficient를 사용합니다.
    따라서 이번 1차 실험의 질문은 하나입니다.

    > 전체 physics-loss 양을 늘리지 않고, 어려운 공간에 loss를 재배치하면 validation residual이 개선되는가?

    `QUICK_MODE=True`는 Colab 실행 경로를 검증하는 smoke test입니다. 성능 결론에는 충분하지 않습니다.
    """
)

cells.extend(
    [
        markdown(
            r"""
            ## 4. Adaptive loss 설계

            기존 PhysicsNeMo `PhysicsInformer`가 만든 Darcy residual $r(x)$를 그대로 사용합니다.
            residual weight는 gradient 경로에서 분리하고 샘플마다 평균 1로 정규화합니다.

            $$
            w(x)=\operatorname{mean1}\left[1+\alpha\,
            \sigma\left(\frac{s(x)-1}{T}\right)\right],\qquad
            s(x)=\frac{|r(x)|-\operatorname{median}(|r|)}{Q_{0.9}(|r|)-\operatorname{median}(|r|)+\epsilon}
            $$

            $$
            L_{adaptive}=L_{data}+\frac{0.1}{240}\operatorname{mean}(w(x)|r(x)|)
            $$

            안전장치:

            - `residual.detach()`로 weight 자체를 통한 우회 gradient 차단
            - 기존 예제와 동일하게 2-cell 테두리 제외
            - 각 sample의 내부 weight 평균을 1로 고정
            - 전역 physics coefficient `0.1/240`은 fixed/adaptive에서 동일
            - `MATERIAL_PRIOR_WEIGHT=0.0`이 기본값: 불연속 $k$에서 strong-form 미분 오차를 곧바로 강화하지 않음
            """
        ),
        code(
            r"""
            # 4-1. 공간 weight와 fixed/adaptive loss
            import copy
            import time

            QUICK_MODE = True  # @param {type:"boolean"}
            ADAPTIVE_ALPHA = 4.0  # @param {type:"number"}
            ADAPTIVE_TEMPERATURE = 0.7  # @param {type:"number"}
            MATERIAL_PRIOR_WEIGHT = 0.0  # @param {type:"number"}

            # QUICK_MODE는 두 모델 각각 12 update만 실행하는 Colab smoke test입니다.
            # False이면 두 모델 각각 공식 train split을 10 epoch 학습합니다.
            STEPS_PER_VARIANT = 12 if QUICK_MODE else len(official_train_dataset) * 10
            VALIDATION_LIMIT = 8 if QUICK_MODE else len(official_valid_dataset)

            PDE_INTERIOR_MASK = torch.zeros(
                1, 1, 240, 240, device=DEVICE, dtype=official_invar.dtype
            )
            PDE_INTERIOR_MASK[:, :, 2:-2, 2:-2] = 1.0


            def material_gradient_magnitude(k_scaled):
                gx = torch.zeros_like(k_scaled)
                gy = torch.zeros_like(k_scaled)
                gx[:, :, 1:-1, :] = (
                    k_scaled[:, :, 2:, :] - k_scaled[:, :, :-2, :]
                ) / (2.0 * OFFICIAL_DX)
                gy[:, :, :, 1:-1] = (
                    k_scaled[:, :, :, 2:] - k_scaled[:, :, :, :-2]
                ) / (2.0 * OFFICIAL_DX)
                return torch.sqrt(gx.square() + gy.square() + 1e-12)


            def robust_sample_score(value):
                interior = value[:, :, 2:-2, 2:-2].flatten(1)
                median = torch.quantile(interior, 0.50, dim=1, keepdim=True)
                q90 = torch.quantile(interior, 0.90, dim=1, keepdim=True)
                scale = (q90 - median).clamp_min(1e-6)
                return (value - median[:, :, None, None]) / scale[:, :, None, None]


            def adaptive_spatial_weights(k_scaled, residual_padded):
                residual_score = robust_sample_score(residual_padded.detach().abs())
                if MATERIAL_PRIOR_WEIGHT > 0.0:
                    material_score = robust_sample_score(
                        material_gradient_magnitude(k_scaled).detach()
                    )
                    score = (
                        (1.0 - MATERIAL_PRIOR_WEIGHT) * residual_score
                        + MATERIAL_PRIOR_WEIGHT * material_score
                    )
                else:
                    score = residual_score

                attention = torch.sigmoid(
                    (score - 1.0) / ADAPTIVE_TEMPERATURE
                )
                weights = (1.0 + ADAPTIVE_ALPHA * attention) * PDE_INTERIOR_MASK
                interior_mean = weights.sum((2, 3), keepdim=True) / PDE_INTERIOR_MASK.sum()
                return (weights / interior_mean.clamp_min(1e-6)).detach()


            def comparison_losses(model, invar, outvar, mode):
                k_scaled = invar[:, 0:1]
                prediction = model(k_scaled)
                residual = official_informer.forward({
                    "u": prediction,
                    "k": k_scaled,
                })["diffusion_u"]
                residual_padded = F.pad(
                    residual[:, :, 2:-2, 2:-2],
                    [2, 2, 2, 2],
                    mode="constant",
                    value=0,
                )
                if mode == "adaptive":
                    weights = adaptive_spatial_weights(k_scaled, residual_padded)
                elif mode == "fixed":
                    weights = PDE_INTERIOR_MASK
                else:
                    raise ValueError(f"unknown mode: {mode}")

                data_loss = F.mse_loss(outvar, prediction)
                # 두 모델에 공통인 평가량: 공간 weight를 적용하지 않은 원본 PDE L1.
                pde_uniform = residual_padded.abs().mean()
                # 학습에 실제 사용되는 PDE objective. fixed에서는 pde_uniform과 정확히 같습니다.
                pde_objective = (
                    weights * residual_padded.abs()
                ).sum() / residual_padded.numel()
                physics_contribution = (
                    OFFICIAL_DX * OFFICIAL_PHYSICS_WEIGHT * pde_objective
                )
                total_loss = data_loss + physics_contribution
                return {
                    "prediction": prediction,
                    "residual": residual_padded,
                    "weights": weights,
                    "data": data_loss,
                    "pde_uniform": pde_uniform,
                    "pde_objective": pde_objective,
                    "physics_contribution": physics_contribution,
                    "total": total_loss,
                }


            print("mode / steps / validation:", QUICK_MODE, STEPS_PER_VARIANT, VALIDATION_LIMIT)
            """
        ),
        markdown(
            r"""
            ### Colab smoke test — forward, backward, 불변조건

            긴 학습 전에 다음을 확인합니다.

            1. fixed loss가 기존 공식 loss와 수치적으로 같은가?
            2. adaptive weight가 `[B,1,240,240]`이고 내부 평균이 1인가?
            3. weight에 gradient가 연결되지 않았는가?
            4. adaptive total loss가 FNO 파라미터까지 정상적으로 backward되는가?
            """
        ),
        code(
            r"""
            # 4-2. 실제 240x240 batch에서 forward/backward smoke test
            official_model.eval()
            with torch.no_grad():
                original_check = official_pino_losses(
                    official_model, official_invar, official_outvar
                )
                fixed_check = comparison_losses(
                    official_model, official_invar, official_outvar, "fixed"
                )
                adaptive_check = comparison_losses(
                    official_model, official_invar, official_outvar, "adaptive"
                )

            interior_weight_mean = (
                adaptive_check["weights"].sum()
                / (official_invar.shape[0] * PDE_INTERIOR_MASK.sum())
            )
            print(
                "original/fixed uniform PDE L1:",
                float(original_check["pde"]),
                float(fixed_check["pde_uniform"]),
            )
            print("adaptive weight mean:", float(interior_weight_mean))
            print("adaptive weight min/max:",
                  float(adaptive_check["weights"][adaptive_check["weights"] > 0].min()),
                  float(adaptive_check["weights"].max()))

            assert torch.allclose(
                original_check["pde"], fixed_check["pde_objective"], atol=1e-6
            )
            assert adaptive_check["weights"].shape == official_outvar.shape
            assert not adaptive_check["weights"].requires_grad
            assert torch.allclose(
                interior_weight_mean,
                torch.tensor(1.0, device=DEVICE),
                atol=1e-5,
            )

            official_model.train()
            official_model.zero_grad(set_to_none=True)
            backward_check = comparison_losses(
                official_model, official_invar, official_outvar, "adaptive"
            )
            backward_check["total"].backward()
            gradient_is_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in official_model.parameters()
            )
            assert gradient_is_finite
            official_model.zero_grad(set_to_none=True)
            print("CHECK PASSED: fixed equivalence + adaptive invariants + backward")
            """
        ),
        markdown(
            r"""
            ## 5. 같은 초기값·같은 순서로 fixed와 adaptive 학습

            QUICK mode의 목적은 실행 검증입니다. 12 step 결과로 우열을 주장하지 않습니다.
            실제 비교 시 `QUICK_MODE=False`로 바꾸고 여러 seed를 반복해야 합니다.
            """
        ),
        code(
            r"""
            # 5-1. 공정한 비교를 위한 초기 state와 deterministic DataLoader
            def build_comparison_model():
                return FNO(
                    in_channels=1,
                    out_channels=1,
                    decoder_layers=1,
                    decoder_layer_size=32,
                    dimension=2,
                    latent_channels=32,
                    num_fno_layers=4,
                    num_fno_modes=12,
                    padding=9,
                ).to(DEVICE)


            comparison_initial_state = {
                key: value.detach().cpu().clone()
                for key, value in official_model.state_dict().items()
            }


            def make_train_loader():
                return DataLoader(
                    official_train_dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                    generator=torch.Generator().manual_seed(SEED),
                )


            def train_variant(mode):
                model = build_comparison_model()
                model.load_state_dict(comparison_initial_state)
                optimizer = torch.optim.Adam(
                    model.parameters(), betas=(0.9, 0.999), lr=0.001, weight_decay=0.0
                )
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.99948708
                )
                loader = make_train_loader()
                iterator = iter(loader)
                history = {
                    "data": [],
                    "pde_uniform": [],
                    "pde_objective": [],
                    "physics_contribution": [],
                    "total": [],
                    "lr": [],
                }
                started = time.perf_counter()

                model.train()
                for step in range(1, STEPS_PER_VARIANT + 1):
                    try:
                        invar, outvar, _, _ = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        invar, outvar, _, _ = next(iterator)

                    optimizer.zero_grad(set_to_none=True)
                    losses = comparison_losses(model, invar, outvar, mode)
                    losses["total"].backward()
                    optimizer.step()
                    scheduler.step()

                    for key in [
                        "data",
                        "pde_uniform",
                        "pde_objective",
                        "physics_contribution",
                        "total",
                    ]:
                        history[key].append(float(losses[key].detach()))
                    history["lr"].append(optimizer.param_groups[0]["lr"])
                    if step == 1 or step % max(1, STEPS_PER_VARIANT // 4) == 0:
                        print(
                            f"{mode:8s} {step:4d}/{STEPS_PER_VARIANT} | "
                            f"data={history['data'][-1]:.4e} | "
                            f"uniform PDE={history['pde_uniform'][-1]:.4e} | "
                            f"PDE objective={history['pde_objective'][-1]:.4e}"
                        )

                print(mode, "elapsed:", f"{time.perf_counter() - started:.1f}s")
                return model, history


            fixed_model, fixed_history = train_variant("fixed")
            adaptive_model, adaptive_history = train_variant("adaptive")
            """
        ),
        markdown(
            r"""
            ### 5-2. 기존 GitHub PINO loss와 adaptive loss를 같은 축에서 비교

            여기서 **Existing PINO**는 이 저장소의 기존
            [`darcy_pino_physicsnemo_colab.ipynb`](darcy_pino_physicsnemo_colab.ipynb)에 있는
            loss를 그대로 사용해 같은 초기값으로 다시 학습한 `fixed` run입니다.
            별도의 50-epoch checkpoint를 불러온 비교는 아닙니다.

            | 표시 이름 | 수식 | 의미 |
            |---|---|---|
            | Data MSE | $L_{data}=\operatorname{mean}((\hat u-u)^2)$ | 정답 pressure와 예측 pressure의 오차 |
            | Uniform PDE L1 | $L_{PDE}^{uniform}=\operatorname{mean}(|r|)$ | **두 모델에 공통으로 적용하는** raw Darcy residual 평가량 |
            | Existing PINO PDE objective | $L_{PDE}^{fixed}=L_{PDE}^{uniform}$ | 기존 GitHub PINO가 학습에 쓰는 PDE loss |
            | Adaptive PINO PDE objective | $L_{PDE}^{adaptive}=\operatorname{mean}(w(x)|r(x)|)$ | 어려운 위치를 강조한 PDE loss, 내부 $\operatorname{mean}(w)=1$ |
            | Physics contribution | $\frac{0.1}{240}L_{PDE}^{objective}$ | total loss에 실제 더해지는 PDE 항 |
            | Total objective | $L_{data}+\frac{0.1}{240}L_{PDE}^{objective}$ | optimizer가 최소화하는 최종 loss |

            따라서 **Uniform PDE L1 그래프가 사과 대 사과 비교**이고,
            PDE objective/physics contribution 그래프는 각 모델이 실제로 무엇을 최적화했는지 보여줍니다.
            아래 곡선은 epoch 평균이 아니라 batch size 1의 optimizer-step 값이라 출렁이는 것이 정상입니다.
            """
        ),
        code(
            r"""
            # 5-2. loss 항별 fixed-vs-adaptive 비교 — 모든 축과 범례를 명시
            comparison_labels = {
                "fixed": "Existing PINO — uniform PDE loss",
                "adaptive": "Adaptive PINO — spatially weighted PDE loss",
            }
            fixed_steps = np.arange(1, len(fixed_history["total"]) + 1)
            adaptive_steps = np.arange(1, len(adaptive_history["total"]) + 1)

            fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            plot_specs = [
                (
                    "data",
                    "A. Pressure data mismatch",
                    "Data MSE: mean((predicted u - target u)^2)",
                ),
                (
                    "pde_uniform",
                    "B. Common physics metric (apples-to-apples)",
                    "Uniform Darcy PDE L1: mean(|residual|)",
                ),
                (
                    "physics_contribution",
                    "C. Physics term actually added to total loss",
                    "Physics contribution: (0.1 / 240) × PDE objective",
                ),
                (
                    "total",
                    "D. Objective minimized by Adam",
                    "Total training objective",
                ),
            ]
            for axis, (key, title, ylabel) in zip(axes.flat, plot_specs):
                axis.semilogy(
                    fixed_steps,
                    fixed_history[key],
                    marker="o",
                    markersize=3,
                    label=comparison_labels["fixed"],
                )
                axis.semilogy(
                    adaptive_steps,
                    adaptive_history[key],
                    marker="s",
                    markersize=3,
                    label=comparison_labels["adaptive"],
                )
                axis.set_title(title)
                axis.set_xlabel("Optimizer step")
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.25, which="both")
                axis.legend(title="Model / training loss", fontsize=8)

            plt.show()
            """
        ),
        markdown(
            r"""
            ## 6. Validation: 전체 오차와 permeability 급변 영역을 분리

            전체 relative L2와 PDE L1 외에 $|\nabla k|$ 상위 10% 영역의 residual을 따로 측정합니다.
            재료 급변도는 여기서 **평가 마스크**로만 사용하며 기본 adaptive loss에는 넣지 않습니다.
            """
        ),
        code(
            r"""
            # 6-1. 동일 validation subset 평가와 시각화
            @torch.no_grad()
            def evaluate_variant(model, mode):
                model.eval()
                metrics = {
                    "data_mse": [],
                    "relative_l2": [],
                    "uniform_pde_l1": [],
                    "interface_uniform_pde_l1": [],
                }
                shown = None
                for index, (invar, outvar, _, _) in enumerate(official_valid_loader):
                    if index >= VALIDATION_LIMIT:
                        break
                    result = comparison_losses(model, invar, outvar, mode)
                    prediction = result["prediction"]
                    residual_abs = result["residual"].abs()
                    gradient = material_gradient_magnitude(invar[:, 0:1]) * PDE_INTERIOR_MASK
                    flat_gradient = gradient[:, :, 2:-2, 2:-2].flatten(1)
                    threshold = torch.quantile(flat_gradient, 0.90, dim=1, keepdim=True)
                    interface_mask = (
                        gradient >= threshold[:, :, None, None]
                    ).to(residual_abs.dtype) * PDE_INTERIOR_MASK

                    metrics["data_mse"].append(float(F.mse_loss(prediction, outvar)))
                    metrics["relative_l2"].append(float(
                        torch.linalg.vector_norm(prediction - outvar)
                        / torch.linalg.vector_norm(outvar).clamp_min(1e-12)
                    ))
                    metrics["uniform_pde_l1"].append(float(residual_abs.mean()))
                    metrics["interface_uniform_pde_l1"].append(float(
                        (residual_abs * interface_mask).sum()
                        / interface_mask.sum().clamp_min(1.0)
                    ))
                    if shown is None:
                        shown = {key: value.detach().cpu() for key, value in {
                            "k": invar[:, 0:1] * 4.49996e00,
                            "target": outvar * 3.88433e-03,
                            "prediction": prediction * 3.88433e-03,
                            "residual": residual_abs,
                            "weights": result["weights"],
                        }.items()}

                return {key: float(np.mean(value)) for key, value in metrics.items()}, shown


            fixed_metrics, fixed_shown = evaluate_variant(fixed_model, "fixed")
            adaptive_metrics, adaptive_shown = evaluate_variant(adaptive_model, "adaptive")
            metric_header = (
                f"{'model':<18} {'data MSE':>12} {'relative L2':>12} "
                f"{'uniform PDE L1':>16} {'interface PDE L1':>18}"
            )
            print(metric_header)
            print("-" * len(metric_header))
            for model_name, metrics in [
                ("Existing PINO", fixed_metrics),
                ("Adaptive PINO", adaptive_metrics),
            ]:
                print(
                    f"{model_name:<18} {metrics['data_mse']:>12.4e} "
                    f"{metrics['relative_l2']:>12.4e} "
                    f"{metrics['uniform_pde_l1']:>16.4e} "
                    f"{metrics['interface_uniform_pde_l1']:>18.4e}"
                )

            fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
            for row, (name, shown) in enumerate([
                ("fixed", fixed_shown), ("adaptive", adaptive_shown)
            ]):
                display_name = "Existing PINO" if name == "fixed" else "Adaptive PINO"
                panels = [
                    (shown["k"][0, 0], "Permeability k", "viridis", "k [physical scale]"),
                    (shown["target"][0, 0], "Target pressure u", "magma", "u [physical scale]"),
                    (shown["prediction"][0, 0], f"{display_name} prediction", "magma", "u [physical scale]"),
                    (shown["residual"][0, 0], f"{display_name} |Darcy residual|", "inferno", "|residual| [scaled PDE units]"),
                    (shown["weights"][0, 0], f"{display_name} spatial weight", "plasma", "w(x), interior mean = 1"),
                ]
                for axis, (image, title, cmap, colorbar_label) in zip(axes[row], panels):
                    rendered = axis.imshow(image.numpy(), origin="lower", cmap=cmap)
                    axis.set_title(title)
                    axis.set_xlabel("x grid index")
                    axis.set_ylabel("y grid index")
                    colorbar = plt.colorbar(rendered, ax=axis, fraction=0.046)
                    colorbar.set_label(colorbar_label)
            plt.show()
            """
        ),
        markdown(
            r"""
            ## 해석 기준과 다음 ablation

            QUICK mode가 통과해도 adaptive 방법이 더 좋다고 결론 내리면 안 됩니다. 다음 순서로 실험을 늘립니다.

            1. 3개 이상 seed에서 fixed/adaptive 반복
            2. validation relative L2, 전체 PDE L1, interface PDE L1의 평균·표준편차 보고
            3. `ADAPTIVE_ALPHA = 1, 2, 4` 비교
            4. 그 뒤에만 `MATERIAL_PRIOR_WEIGHT > 0` 실험
            5. material prior를 켤 때는 flux-conservative residual과 PhysicsInformer residual을 별도 비교
            """
        ),
        code(
            r"""
            # 6-2. 두 모델과 실험 설정을 하나의 checkpoint로 저장
            adaptive_checkpoint_path = Path(
                "/content/darcy_adaptive_pino_physicsnemo_2_1_1.pt"
            )
            torch.save({
                "physicsnemo_version": physicsnemo.__version__,
                "source": "NVIDIA Darcy 240x240 + residual-driven spatial weighting",
                "quick_mode": QUICK_MODE,
                "steps_per_variant": STEPS_PER_VARIANT,
                "adaptive_alpha": ADAPTIVE_ALPHA,
                "adaptive_temperature": ADAPTIVE_TEMPERATURE,
                "material_prior_weight": MATERIAL_PRIOR_WEIGHT,
                "fixed_state_dict": fixed_model.state_dict(),
                "adaptive_state_dict": adaptive_model.state_dict(),
                "fixed_history": fixed_history,
                "adaptive_history": adaptive_history,
                "fixed_metrics": fixed_metrics,
                "adaptive_metrics": adaptive_metrics,
            }, adaptive_checkpoint_path)
            print("saved:", adaptive_checkpoint_path)
            print("size :", f"{adaptive_checkpoint_path.stat().st_size / 1024**2:.2f} MB")
            """
        ),
    ]
)

notebook = {
    **baseline,
    "cells": cells,
    "metadata": {
        **baseline.get("metadata", {}),
        "accelerator": "GPU",
        "colab": {
            "name": OUTPUT.name,
            "provenance": [],
        },
    },
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
export_percent_script(cells)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
print(f"Wrote {SCRIPT}")

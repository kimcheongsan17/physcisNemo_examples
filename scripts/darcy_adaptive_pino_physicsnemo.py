# -*- coding: utf-8 -*-
"""Companion script for the adaptive Darcy PINO Colab notebook."""

# %% [markdown]
# # Adaptive Darcy PINO on PhysicsNeMo — fixed vs spatially weighted physics loss
#
# 이 노트북은 기존 [`darcy_pino_physicsnemo_colab.ipynb`](darcy_pino_physicsnemo_colab.ipynb)의
# NVIDIA 공식 240×240 Darcy 설정을 그대로 준비한 뒤, **physics loss의 공간 가중치만** 바꾸어 비교합니다.
#
# - `fixed PINO`: 모든 내부 pixel의 PDE residual을 같은 비중으로 학습
# - `adaptive PINO`: 현재 residual이 큰 내부 pixel을 더 강하게 학습
#
# 두 run은 같은 FNO 초기값, 같은 데이터 순서, 같은 optimizer와 같은 전역 physics coefficient를 사용합니다.
# 따라서 이번 1차 실험의 질문은 하나입니다.
#
# > 전체 physics-loss 양을 늘리지 않고, 어려운 공간에 loss를 재배치하면 validation residual이 개선되는가?
#
# `QUICK_MODE=True`는 Colab 실행 경로를 검증하는 smoke test입니다. 성능 결론에는 충분하지 않습니다.

# %% [markdown]
# ## 1. 네트워크와 loss의 전체 흐름
#
# ```text
# permeability k(x,y)  [B, 1, H, W]
#             │
#             ▼
# PhysicsNeMo FNO
#   ├─ lifting / coordinate features
#   ├─ [Fourier spectral convolution + local 1x1 path + GELU] × 4
#   └─ point-wise decoder MLP
#             │
#             ▼
# predicted solution û(x,y)  [B, 1, H, W]
#       ┌─────┴──────────────────────┐
#       ▼                            ▼
# data loss                     PhysicsInformer
# MSE(û, u)               -div(k grad(û)) - 1
#       └────────────┬───────────────┘
#                    ▼
#   total loss = data loss + λ · dx · PDE loss
# ```
#
# FNO의 spectral convolution은 FFT로 field를 Fourier 공간으로 옮긴 뒤 일부 mode를 학습합니다.
# 그래서 한 지점의 출력이 주변 몇 pixel뿐 아니라 **영역 전체의 패턴**을 볼 수 있습니다.
# `PhysicsInformer`는 같은 예측 field에 finite difference를 적용해 PDE 잔차를 만들며, 이 계산도 PyTorch graph 안에 있으므로 FNO까지 gradient가 전달됩니다.

# %% [markdown]
# ## 이 노트북의 셀 설명을 읽는 법
#
# 각 실행 셀 바로 앞의 **셀 해설**은 다음 네 가지를 구분해 적습니다.
#
# 1. **입력/출력 데이터 구조**: `B`=batch, `C`=channel, `H/W`=공간 격자입니다.
# 2. **처리 주체**: 파일 처리인지, FNO 신경망인지, `PhysicsInformer`의 수치 미분인지 표시합니다.
# 3. **loss와 gradient**: 해당 셀이 loss를 정의하거나 optimizer로 가중치를 바꾸는지 표시합니다.
# 4. **확인 포인트**: 실행 뒤 어떤 출력과 shape가 정상인지 적습니다.
#
# ### 전체 텐서 흐름
#
# ```text
# HDF5 원본
#   Kcoeff, sol: [N, 1, 241, 241]
#          │ 공식 utils.py의 경계 crop [:, :, :240, :240] + scale
#          ▼
#   k, u: [B, 1, 240, 240]
#          │
#          ├──────────────► data loss = MSE(u_hat, u)
#          │
#          ▼
# FNO(k) = u_hat: [B, 1, 240, 240]
#   ├─ 내부 좌표 feature (x,y) 추가
#   ├─ latent channel 32
#   ├─ Fourier mode 12, spectral layer 4개, padding 9
#   └─ point-wise decoder → pressure 1 channel
#          │
#          ▼
# PhysicsInformer(k, u_hat)
#   finite difference로 -div(k grad(u_hat)) - Q 계산
#          │
#          ▼
# PDE residual: [B, 1, 240, 240]
#
# total loss = MSE(u_hat, u) + (1/240) × 0.1 × mean(abs(PDE residual))
# ```
#
# 여기서 PINO는 **FNO 뒤에 별도의 PINN 네트워크를 연결한 모델이 아닙니다.**
# 학습되는 네트워크는 FNO 하나이고, `PhysicsInformer`가 만든 미분 가능한 PDE loss가 FNO의 파라미터까지 역전파됩니다.

# %% [markdown]
# ## PhysicsNeMo에서 무엇을 가져오고, 이 Colab은 무엇을 추가하나
#
# 이 실습은 NVIDIA 코드를 한 덩어리로 복사한 것이 아니라 **설치 패키지**와 **공식 예제 파일**을 역할별로 연결합니다.
#
# ```text
# pip: nvidia-physicsnemo[sym]==2.1.1
#   ├─ physicsnemo.models.fno.FNO
#   │    └─ 실제로 학습되는 neural operator
#   └─ physicsnemo.sym.eq.phy_informer.PhysicsInformer
#        └─ PDE 미분과 residual 계산 그래프
#
# NVIDIA GitHub 공식 Darcy 예제 (고정 commit)
#   ├─ utils.py
#   │    ├─ HDF5MapStyleDataset: crop/scale/DataLoader sample 생성
#   │    └─ Diffusion: Darcy PDE의 symbolic equation 정의
#   ├─ conf/config_pino.yaml: mode/layer/padding/loss weight 설정의 근거
#   └─ darcy_physics_informed_fno.py: loss와 explicit training loop의 근거
#
# 이 Colab에서 추가한 부분
#   ├─ 셀별 한국어 설명과 tensor shape 표
#   ├─ T4에서 보이는 실시간 plot
#   ├─ assert/NaN/shape/API 출처 디버깅
#   └─ 전체 validation 지표와 checkpoint 묶음 저장
# ```
#
# | 구성요소 | 실제 import/파일 | 네트워크 처리에서의 역할 | 학습 파라미터 |
# |---|---|---|---|
# | FNO | `from physicsnemo.models.fno import FNO` | `k → u_hat` field-to-field mapping | 있음 |
# | PhysicsInformer | `from physicsnemo.sym.eq.phy_informer import PhysicsInformer` | `k, u_hat → PDE residual` | 없음 |
# | Diffusion | 공식 예제 `utils.py` | Darcy symbolic PDE 정의 | 없음 |
# | HDF5MapStyleDataset | 공식 예제 `utils.py` | 241→240 crop, scaling, tensor 생성 | 없음 |
# | Adam/ExponentialLR | PyTorch | FNO 파라미터 업데이트 | optimizer state만 있음 |
# | plot/checkpoint | 이 Colab + Matplotlib/PyTorch | 관찰·저장 | 없음 |
#
# 따라서 “PhysicsNeMo로 네트워크 처리한다”는 핵심 호출은 `official_model(k_scaled)`이고,
# “PhysicsNeMo로 물리를 연결한다”는 핵심 호출은 `official_informer.forward({"u": prediction, "k": k_scaled})`입니다.

# %% [markdown]
# ## 2. Colab 설치
#
# 현재 공식 `physicsnemo.sym`/`PhysicsInformer`가 포함된 PhysicsNeMo 2.1.1로 고정합니다.
# 설치가 PyTorch를 갱신했다는 경고를 내면 이 셀 실행 후 **런타임 → 세션 다시 시작**을 한 번 선택하고, 다시 위에서 실행하세요.

# %% [markdown]
# ### 셀 해설 — PhysicsNeMo 실행 환경 설치
#
# | 구분 | 내용 |
# |---|---|
# | 입력 | Colab의 Python 버전과 현재 설치된 패키지 메타데이터 |
# | 처리 | `nvidia-physicsnemo[sym]==2.1.1`, Matplotlib, SciPy 설치 여부 확인 |
# | 출력 | 이후 셀에서 사용할 `physicsnemo.models.fno.FNO`와 `PhysicsInformer` |
# | 텐서/네트워크/loss | 아직 생성하지 않음 |
#
# 이 셀은 모델 계산이 아니라 **재현 가능한 소프트웨어 환경**을 고정합니다. 설치 과정에서 PyTorch가 바뀌었다면 런타임을 다시 시작한 뒤 처음부터 실행합니다.
#
# #### PhysicsNeMo 연결
#
# 이 셀은 PyPI의 `nvidia-physicsnemo[sym]==2.1.1`을 설치합니다. 이후 패키지에서 FNO와 PhysicsInformer를 import할 수 있게 만드는 **라이브러리 공급 단계**이며, 아직 neural network forward는 실행하지 않습니다.

# %%
# 2-1. 필요한 패키지만 설치합니다. 이미 정확한 버전이면 설치를 건너뜁니다.
import importlib.util
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

PHYSICSNEMO_VERSION = "2.1.1"

if sys.version_info < (3, 11):
    raise RuntimeError(
        f"PhysicsNeMo {PHYSICSNEMO_VERSION}은 Python 3.11 이상이 필요합니다. "
        "Colab의 최신 런타임을 선택해 주세요."
    )

try:
    installed_version = version("nvidia-physicsnemo")
except PackageNotFoundError:
    installed_version = None

needs_install = (
    installed_version != PHYSICSNEMO_VERSION
    or importlib.util.find_spec("sympy") is None
)

if needs_install:
    print(f"Installing NVIDIA PhysicsNeMo {PHYSICSNEMO_VERSION} ...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            f"nvidia-physicsnemo[sym]=={PHYSICSNEMO_VERSION}",
            "matplotlib>=3.10.8",
            "scipy",
        ]
    )
else:
    print(f"PhysicsNeMo {installed_version} is already installed.")

print("설치 셀 완료")

# %% [markdown]
# ### 셀 해설 — 난수·장치·버전 진단
#
# | 구분 | 내용 |
# |---|---|
# | 입력 | Python/PhysicsNeMo/PyTorch 런타임 |
# | 처리 | `SEED=42` 고정, CUDA 사용 가능 여부 확인, T4에서는 matmul 정밀도 설정 |
# | 출력 | 이후 모든 텐서와 FNO가 올라갈 `DEVICE` |
# | 텐서/네트워크/loss | 아직 생성하지 않음 |
#
# 같은 seed는 데이터 순서와 초기 가중치의 변동을 줄여 비교를 쉽게 합니다. GPU가 아니면 240×240 FNO 학습이 매우 느리므로 `device: cuda`와 T4 이름을 확인합니다.
#
# #### PhysicsNeMo 연결
#
# `import physicsnemo`로 실제 설치 버전을 확인합니다. PhysicsNeMo의 `FNO(...).to(DEVICE)`와 PhysicsInformer가 모두 이 셀의 `DEVICE`를 공유하므로 입력 tensor와 모델이 같은 CUDA 장치에 있어야 합니다.

# %%
# 2-2. 런타임, GPU, 패키지 버전을 진단합니다.
import random
import time

import matplotlib
import numpy as np
import physicsnemo
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

print("python       :", sys.version.split()[0])
print("physicsnemo  :", physicsnemo.__version__)
print("torch        :", torch.__version__)
print("matplotlib   :", matplotlib.__version__)
print("device       :", DEVICE)
print(
    "accelerator  :",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback",
)

if DEVICE.type != "cuda":
    print("WARNING: GPU가 없어 자동으로 작은 CPU 설정을 사용합니다.")
    print("런타임 > 런타임 유형 변경 > T4 GPU를 선택하면 훨씬 빠릅니다.")

# %% [markdown]
# ## 3. NVIDIA 원본 예제 스냅샷 받기
#
# 아래 셀은 전체 저장소를 clone하지 않고, 이 실습의 근거가 된 원본 파일 4개만 `/content/nvidia_darcy_pino_reference`에 저장합니다.
# `main`이 나중에 바뀌어도 같은 코드를 받을 수 있도록 2026-06-27에 확인한 commit SHA를 사용합니다.
# 실습 코드는 셀에 풀어 썼기 때문에 네트워크가 끊겨 이 셀이 실패해도 나머지는 실행할 수 있습니다.

# %% [markdown]
# ### 셀 해설 — NVIDIA 원본 코드 스냅샷
#
# | 구분 | 내용 |
# |---|---|
# | 입력 | 고정된 NVIDIA PhysicsNeMo commit의 원본 파일 4개 |
# | 처리 | 원본 Python/YAML/README를 `/content/nvidia_darcy_pino_reference`에 저장 |
# | 출력 | 뒤 셀에서 실제로 import할 공식 `utils.py`와 비교 가능한 설정 파일 |
# | 텐서/네트워크/loss | 아직 생성하지 않음 |
#
# commit SHA를 고정했기 때문에 NVIDIA의 `main` 브랜치가 나중에 바뀌어도 이 실습의 데이터 전처리와 학습식은 동일하게 유지됩니다.
#
# #### PhysicsNeMo 연결
#
# 이 셀은 설치 패키지가 아니라 NVIDIA GitHub의 **공식 Darcy 예제 레이어**를 받습니다. 특히 `utils.py`의 dataset/PDE 클래스와 `config_pino.yaml`의 하이퍼파라미터를 뒤 셀에서 설치된 PhysicsNeMo FNO·PhysicsInformer에 연결합니다.

# %%
# 3-1. 공식 Apache-2.0 예제 파일을 비교·공부용으로 내려받습니다.
from pathlib import Path
from urllib.request import urlopen

NVIDIA_COMMIT = "1d8e2be43655ccaf13979289080f59510fb10648"
NVIDIA_EXAMPLE = (
    f"https://raw.githubusercontent.com/NVIDIA/physicsnemo/{NVIDIA_COMMIT}/"
    "examples/cfd/darcy_physics_informed"
)
reference_dir = Path("/content/nvidia_darcy_pino_reference")
reference_dir.mkdir(parents=True, exist_ok=True)

reference_files = [
    "README.md",
    "darcy_physics_informed_fno.py",
    "utils.py",
    "conf/config_pino.yaml",
]

try:
    for relative_path in reference_files:
        destination = reference_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(urlopen(f"{NVIDIA_EXAMPLE}/{relative_path}").read())
        print("downloaded:", destination)
except Exception as exc:
    print("원본 파일 다운로드를 건너뜁니다:", repr(exc))

original_script = reference_dir / "darcy_physics_informed_fno.py"
if original_script.exists():
    print("\n--- NVIDIA 원본 학습 스크립트 앞부분 ---")
    print("".join(original_script.read_text(encoding="utf-8").splitlines(True)[:55]))

# %% [markdown]
# ### 셀 해설 — 공식 Darcy HDF5 다운로드와 분할
#
# | 구분 | 내용 |
# |---|---|
# | 원본 key | `Kcoeff`=permeability, `sol`=pressure solution |
# | 원본 shape | 각각 `[1024, 1, 241, 241]` |
# | 분할 | 앞 10%=train 102개, 다음 10%=validation 102개 |
# | 출력 파일 | `train.hdf5`, `validation.hdf5` |
# | 네트워크/loss | 아직 사용하지 않음 |
#
# 이 단계는 **공간 해상도를 줄이지 않습니다.** 241×241 원본을 그대로 HDF5에 보존합니다. 다음 셀에서 NVIDIA 공식 dataset 클래스가 경계 한 줄만 잘라 240×240으로 만듭니다.
#
# #### PhysicsNeMo 연결
#
# 다운로드/분할 규칙은 NVIDIA 공식 `download_data.py` 흐름을 따르지만 `gdown`, `h5py`, SciPy가 파일 I/O를 담당합니다. 즉 이 셀은 PhysicsNeMo 신경망 API가 아니라 **공식 예제와 같은 입력 데이터**를 준비하는 어댑터입니다.

# %%
# 13-2. NVIDIA 공식 download_data.py와 동일한 Darcy_241 다운로드/분할
# Google Drive ID와 10% train + 10% validation 분할을 공식 코드 그대로 사용합니다.
import gdown
import h5py
import numpy as np
import scipy.io
import zipfile

OFFICIAL_DATA_DIR = Path("/content/Darcy_241")
OFFICIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
OFFICIAL_ZIP = Path("/content/Darcy_241.zip")
OFFICIAL_TRAIN_H5 = OFFICIAL_DATA_DIR / "train.hdf5"
OFFICIAL_VALID_H5 = OFFICIAL_DATA_DIR / "validation.hdf5"
DATASET_FILE = OFFICIAL_DATA_DIR / "piececonst_r241_N1024_smooth1.hdf5"

if not DATASET_FILE.exists():
    if not OFFICIAL_ZIP.exists():
        print("공식 Darcy_241 원본 다운로드 중...")
        gdown.download(
            id="1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV",
            output=str(OFFICIAL_ZIP),
            quiet=False,
        )
    print("압축 해제 중...")
    with zipfile.ZipFile(OFFICIAL_ZIP, "r") as archive:
        archive.extractall(OFFICIAL_DATA_DIR)

    # 공식 zip이 .mat 형식이면 NVIDIA utils.py와 동일하게 HDF5로 변환합니다.
    for mat_path in OFFICIAL_DATA_DIR.rglob("*.mat"):
        h5_path = mat_path.with_suffix(".hdf5")
        if not h5_path.exists():
            print("HDF5 변환:", mat_path.name)
            mat_data = scipy.io.loadmat(mat_path)
            with h5py.File(h5_path, "w") as h5_file:
                for key in [k for k in mat_data if not k.startswith("__")]:
                    h5_file.create_dataset(
                        key,
                        data=np.expand_dims(mat_data[key], axis=1),
                        dtype="float32",
                    )

    matches = list(OFFICIAL_DATA_DIR.rglob("piececonst_r241_N1024_smooth1.hdf5"))
    if not matches:
        raise FileNotFoundError("piececonst_r241_N1024_smooth1.hdf5를 찾지 못했습니다.")
    DATASET_FILE = matches[0]

if not (OFFICIAL_TRAIN_H5.exists() and OFFICIAL_VALID_H5.exists()):
    print("공식 10%/10% train-validation 분할 생성 중...")
    with h5py.File(DATASET_FILE, "r") as source,          h5py.File(OFFICIAL_TRAIN_H5, "w") as train_file,          h5py.File(OFFICIAL_VALID_H5, "w") as valid_file:
        for key in source.keys():
            data = source[key][:]
            split = int(len(data) * 0.10)
            train_file.create_dataset(key, data=data[:split])
            valid_file.create_dataset(key, data=data[split:2 * split])

with h5py.File(OFFICIAL_TRAIN_H5, "r") as check_file:
    print("keys            :", list(check_file.keys()))
    print("raw Kcoeff shape:", check_file["Kcoeff"].shape)
    print("raw sol shape   :", check_file["sol"].shape)

print("OFFICIAL DATA READY:", DATASET_FILE)

# %% [markdown]
# ### 셀 해설 — Dataset/DataLoader와 240×240 전처리
#
# | 객체 | shape | 의미 |
# |---|---:|---|
# | `official_invar[:, 0:1]` | `[B, 1, 240, 240]` | FNO 입력 permeability `k` |
# | `official_outvar` | `[B, 1, 240, 240]` | 정답 pressure `u` |
# | `official_x`, `official_y` | 공간 좌표 grid | 시각화/좌표 확인용 |
# | batch | `B=1` | T4 메모리에 맞춘 공식 설정 |
#
# 공식 `HDF5MapStyleDataset`은 `241×241 → 240×240`으로 **경계 한 줄 crop**을 적용합니다. interpolation/downsampling이 아닙니다.
# 학습 안정화를 위해 `k_scaled = k_raw / 4.49996`, `u_scaled = u_raw / 3.88433e-3`를 사용하고, 그림에서는 이 scale을 다시 곱혀 물리값으로 복원합니다.
#
# 아직 FNO forward나 loss 계산은 하지 않으며, 이 셀의 assert가 원본 해상도가 유지되었는지 막아 줍니다.
#
# #### PhysicsNeMo 연결
#
# 핵심 연결은 `OfficialHDF5Dataset = official_darcy_utils.HDF5MapStyleDataset`입니다. 이 클래스는 앞에서 받은 NVIDIA 예제 `utils.py`에서 동적으로 import되며, 반환한 `official_invar[:, 0:1]`이 그대로 PhysicsNeMo `FNO.forward()`의 입력이 됩니다.

# %%
# 13-3. NVIDIA 공식 HDF5MapStyleDataset: 241 원본 -> 공식 코드의 240 x 240 crop
# 다운샘플링이 아니라 원본 예제에서 사용하는 경계 한 줄 crop(:240, :240)입니다.
import importlib.util
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

utils_matches = list(Path("/content/nvidia_darcy_pino_reference").rglob("utils.py"))
if not utils_matches:
    raise FileNotFoundError("NVIDIA 공식 utils.py를 찾지 못했습니다.")

spec = importlib.util.spec_from_file_location("official_darcy_utils", utils_matches[0])
if spec is None or spec.loader is None:
    raise ImportError(f"공식 utils.py를 import할 수 없습니다: {utils_matches[0]}")
official_darcy_utils = importlib.util.module_from_spec(spec)
# inspect가 동적 module의 실제 source file을 찾도록 먼저 등록합니다.
sys.modules[spec.name] = official_darcy_utils
spec.loader.exec_module(official_darcy_utils)

OfficialHDF5Dataset = official_darcy_utils.HDF5MapStyleDataset
OfficialDiffusion = official_darcy_utils.Diffusion

official_train_dataset = OfficialHDF5Dataset(OFFICIAL_TRAIN_H5, device=DEVICE)
official_valid_dataset = OfficialHDF5Dataset(OFFICIAL_VALID_H5, device=DEVICE)
official_train_loader = DataLoader(official_train_dataset, batch_size=1, shuffle=True)
official_valid_loader = DataLoader(official_valid_dataset, batch_size=1, shuffle=False)

official_invar, official_outvar, official_x, official_y = next(iter(official_train_loader))

# 공식 utils.py가 적용한 scale을 되돌려 물리값으로 표시합니다.
official_k_raw = official_invar[0, 0].detach().cpu().numpy() * 4.49996e00
official_u_raw = official_outvar[0, 0].detach().cpu().numpy() * 3.88433e-03

print("raw file shape       : (1, 241, 241)")
print("model permeability   :", tuple(official_invar[:, 0:1].shape))
print("model target solution:", tuple(official_outvar.shape))
print("coordinate grids     :", tuple(official_x.shape), tuple(official_y.shape))
print("train / valid count  :", len(official_train_dataset), "/", len(official_valid_dataset))
print("raw k min / max      :", float(official_k_raw.min()), "/", float(official_k_raw.max()))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
im0 = axes[0].imshow(official_k_raw, origin="lower", cmap="viridis")
axes[0].set_title("OFFICIAL permeability k — 240 x 240")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(official_u_raw, origin="lower", cmap="magma")
axes[1].set_title("OFFICIAL target u — 240 x 240")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

for axis in axes:
    axis.set_xlabel("x index (0..239)")
    axis.set_ylabel("y index (0..239)")
plt.show()

assert official_invar[:, 0:1].shape[-2:] == (240, 240)
assert official_outvar.shape[-2:] == (240, 240)
print("CHECK PASSED: NVIDIA 원본 학습 해상도 240 x 240")

# %% [markdown]
# ### 셀 해설 — FNO 네트워크와 PINO loss 정의
#
# #### 1) 학습되는 네트워크: FNO
#
# ```text
# k_scaled [B,1,240,240]
#   → 내부 coordinate feature (x,y) 추가
#   → lifting: latent channel 32
#   → [2D Fourier spectral convolution, mode 12 + local path + GELU] × 4
#   → domain padding 9 제거
#   → point-wise decoder (hidden size 32, 1 layer)
#   → u_hat [B,1,240,240]
# ```
#
# Fourier layer는 FFT 공간에서 저주파 mode 12개를 학습하므로 한 pixel 주변만 보는 CNN과 달리 영역 전체의 상호작용을 다룰 수 있습니다.
#
# #### 2) 학습되지 않는 물리 계산기: PhysicsInformer
#
# `PhysicsInformer`는 별도 신경망이 아니라 finite difference 연산으로 다음 residual을 만듭니다.
#
# $$r = -\nabla\cdot(k\nabla \hat{u}) - Q$$
#
# scale된 `k`, `u`를 사용하므로 `Q = 1 × 4.49996 × 3.88433×10^{-3}`로 맞춥니다. 2-pixel 경계는 PDE penalty에서 제외합니다.
#
# #### 3) 실제 코드의 loss
#
# $$L_{data}=\operatorname{mean}((\hat{u}-u)^2)$$
#
# $$L_{pde}=\operatorname{mean}(|r|)$$
#
# $$L_{total}=L_{data}+\underbrace{(1/240)\times0.1}_{\text{PDE weight}}L_{pde}$$
#
# `L_total.backward()`를 호출하면 data 경로와 PDE 경로의 gradient가 합쳐져 **같은 FNO 파라미터**를 업데이트합니다.
#
# #### PhysicsNeMo 연결
#
# - `FNO`: 설치된 PhysicsNeMo Core 모델. 학습되는 파라미터는 여기에만 있습니다.
# - `PhysicsInformer`: 설치된 PhysicsNeMo Sym 유틸리티. symbolic PDE에 필요한 1·2차 공간 미분을 finite difference로 계산합니다.
# - `OfficialDiffusion`: NVIDIA Darcy 예제 `utils.py`가 제공하는 PDE 정의입니다.
#
# 연결 순서는 `prediction = official_model(k_scaled)` → `official_informer.forward({"u": prediction, "k": k_scaled})`입니다. 두 번째 호출도 PyTorch 연산 그래프에 남기 때문에 PDE loss의 gradient가 첫 번째 FNO 호출까지 전달됩니다.

# %%
# 13-4. NVIDIA v2.1.1 config_pino.yaml / darcy_physics_informed_fno.py 그대로 구성
import torch.nn.functional as F
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

OFFICIAL_PHYSICS_WEIGHT = 0.1
OFFICIAL_DX = 1.0 / 240.0
OFFICIAL_FORCING = 1.0 * 4.49996e00 * 3.88433e-03

official_pde = OfficialDiffusion(
    T="u", time=False, dim=2, D="k", Q=OFFICIAL_FORCING
)
official_informer = PhysicsInformer(
    required_outputs=["diffusion_u"],
    equations=official_pde,
    grad_method="finite_difference",
    device=DEVICE,
    fd_dx=OFFICIAL_DX,
)

official_model = FNO(
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

def official_pino_losses(model, invar, outvar):
    # 공식 코드는 Kcoeff/4.49996 및 sol/3.88433e-3 scale에서 학습합니다.
    k_scaled = invar[:, 0:1]
    prediction = model(k_scaled)
    residual = official_informer.forward({
        "u": prediction,
        "k": k_scaled,
    })["diffusion_u"]

    # NVIDIA 원본과 동일: 2-cell 테두리를 제외한 뒤 0으로 pad합니다.
    residual_padded = F.pad(
        residual[:, :, 2:-2, 2:-2],
        [2, 2, 2, 2],
        mode="constant",
        value=0,
    )
    data_loss = F.mse_loss(outvar, prediction)
    pde_loss = F.l1_loss(residual_padded, torch.zeros_like(residual_padded))
    total_loss = data_loss + OFFICIAL_DX * OFFICIAL_PHYSICS_WEIGHT * pde_loss
    return {
        "prediction": prediction,
        "residual": residual,
        "data": data_loss,
        "pde": pde_loss,
        "total": total_loss,
    }

official_model.eval()
with torch.no_grad():
    official_debug = official_pino_losses(
        official_model,
        official_invar,
        official_outvar,
    )

print("model input shape :", tuple(official_invar[:, 0:1].shape))
print("prediction shape  :", tuple(official_debug["prediction"].shape))
print("PDE residual shape:", tuple(official_debug["residual"].shape))
print("model parameters  :", sum(p.numel() for p in official_model.parameters()))
print("initial data loss :", float(official_debug["data"]))
print("initial PDE L1    :", float(official_debug["pde"]))
print("initial total loss:", float(official_debug["total"]))

assert official_debug["prediction"].shape == (1, 1, 240, 240)
assert torch.isfinite(official_debug["total"])
print("CHECK PASSED: 원본 240 x 240 FNO + PINO forward")

# %% [markdown]
# ### 디버깅 셀 — 어떤 클래스가 PhysicsNeMo에서 왔는지 직접 확인
#
# 아래 셀은 객체의 `__module__`, 실제 source file, constructor signature와 완성된 `official_model` 구조를 출력합니다.
# 설명만 믿는 대신 Colab 런타임에서 다음 연결을 직접 검증합니다.
#
# - FNO/PhysicsInformer의 module 경로가 `physicsnemo...`로 시작하는가?
# - Dataset/Diffusion의 source가 내려받은 NVIDIA 공식 `utils.py`인가?
# - 최종 모델에 Fourier layer 4개와 decoder가 실제로 들어 있는가?

# %%
# PhysicsNeMo 패키지 API와 NVIDIA 예제 클래스의 실제 출처를 런타임에서 확인합니다.
import inspect
import sys

def resolve_source_file(obj, fallback=None):
    """일반 import와 spec 기반 동적 import 모두에서 source 경로를 찾습니다."""
    module_name = getattr(obj, "__module__", type(obj).__module__)
    module_object = sys.modules.get(module_name)
    source_file = getattr(module_object, "__file__", None)

    if source_file is None:
        try:
            source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        except (TypeError, OSError):
            source_file = None

    if source_file is None and fallback is not None:
        source_file = fallback

    return Path(source_file).resolve() if source_file is not None else None

def report_api_origin(name, obj, fallback=None):
    module_name = getattr(obj, "__module__", type(obj).__module__)
    source_path = resolve_source_file(obj, fallback=fallback)
    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature = "<signature unavailable>"
    print(f"[{name}]")
    print("  module   :", module_name)
    print("  source   :", source_path if source_path is not None else "<source unavailable>")
    print("  signature:", signature[:240])
    return source_path

origin_paths = {
    "FNO": report_api_origin("PhysicsNeMo FNO", FNO),
    "PhysicsInformer": report_api_origin(
        "PhysicsNeMo PhysicsInformer", PhysicsInformer
    ),
    "HDF5MapStyleDataset": report_api_origin(
        "NVIDIA example HDF5MapStyleDataset",
        OfficialHDF5Dataset,
        fallback=utils_matches[0],
    ),
    "Diffusion": report_api_origin(
        "NVIDIA example Diffusion",
        OfficialDiffusion,
        fallback=utils_matches[0],
    ),
}

print("\n[instantiated PhysicsNeMo network]")
print(official_model)
print("trainable parameters:", sum(p.numel() for p in official_model.parameters()))
print("reference utils.py   :", utils_matches[0])
print("reference config     :", reference_dir / "conf/config_pino.yaml")
print("reference train code :", reference_dir / "darcy_physics_informed_fno.py")

expected_utils_path = utils_matches[0].resolve()
assert FNO.__module__.startswith("physicsnemo")
assert PhysicsInformer.__module__.startswith("physicsnemo")
assert OfficialHDF5Dataset.__module__ == "official_darcy_utils"
assert OfficialDiffusion.__module__ == "official_darcy_utils"
assert origin_paths["HDF5MapStyleDataset"] == expected_utils_path
assert origin_paths["Diffusion"] == expected_utils_path
assert all(path is not None for path in origin_paths.values())
print("CHECK PASSED: PhysicsNeMo package와 NVIDIA 공식 예제의 연결 출처가 확인됐습니다.")

# %% [markdown]
# ### 디버깅 셀 — 실제 tensor shape, dtype, device 확인
#
# 아래 셀은 새 학습을 하지 않고 현재 batch를 한 번 forward하여 데이터→FNO→PDE residual 흐름을 표로 출력합니다.
# 모든 tensor가 `[1,1,240,240]`, `float32`, 같은 GPU 장치이고 NaN/Inf가 없어야 정상입니다.

# %%
# 네트워크를 업데이트하지 않는 구조/shape 디버깅 셀
def report_tensor(name, tensor):
    tensor_detached = tensor.detach()
    print(
        f"{name:24s} shape={str(tuple(tensor_detached.shape)):20s} "
        f"dtype={str(tensor_detached.dtype):14s} device={str(tensor_detached.device):6s} "
        f"finite={bool(torch.isfinite(tensor_detached).all())} "
        f"min={float(tensor_detached.min()): .3e} max={float(tensor_detached.max()): .3e}"
    )

official_model.eval()
with torch.no_grad():
    debug_flow = official_pino_losses(
        official_model,
        official_invar,
        official_outvar,
    )

debug_residual_padded = F.pad(
    debug_flow["residual"][:, :, 2:-2, 2:-2],
    [2, 2, 2, 2],
    mode="constant",
    value=0,
)

report_tensor("input k_scaled", official_invar[:, 0:1])
report_tensor("target u_scaled", official_outvar)
report_tensor("FNO prediction", debug_flow["prediction"])
report_tensor("raw PDE residual", debug_flow["residual"])
report_tensor("PDE residual padded", debug_residual_padded)

print("\nloss decomposition")
print("  data MSE             =", float(debug_flow["data"]))
print("  PDE L1               =", float(debug_flow["pde"]))
print("  dx * lambda * PDE L1 =", float(OFFICIAL_DX * OFFICIAL_PHYSICS_WEIGHT * debug_flow["pde"]))
print("  total                =", float(debug_flow["total"]))

expected_shape = (1, 1, 240, 240)
assert tuple(official_invar[:, 0:1].shape) == expected_shape
assert tuple(official_outvar.shape) == expected_shape
assert tuple(debug_flow["prediction"].shape) == expected_shape
assert tuple(debug_residual_padded.shape) == expected_shape
assert torch.isfinite(debug_flow["total"])
print("CHECK PASSED: data -> FNO -> PhysicsInformer -> loss 구조가 정상입니다.")

# %% [markdown]
# ## 4. Adaptive loss 설계
#
# 기존 PhysicsNeMo `PhysicsInformer`가 만든 Darcy residual $r(x)$를 그대로 사용합니다.
# residual weight는 gradient 경로에서 분리하고 샘플마다 평균 1로 정규화합니다.
#
# $$
# w(x)=\operatorname{mean1}\left[1+\alpha\,
# \sigma\left(\frac{s(x)-1}{T}\right)\right],\qquad
# s(x)=\frac{|r(x)|-\operatorname{median}(|r|)}{Q_{0.9}(|r|)-\operatorname{median}(|r|)+\epsilon}
# $$
#
# $$
# L_{adaptive}=L_{data}+\frac{0.1}{240}\operatorname{mean}(w(x)|r(x)|)
# $$
#
# 안전장치:
#
# - `residual.detach()`로 weight 자체를 통한 우회 gradient 차단
# - 기존 예제와 동일하게 2-cell 테두리 제외
# - 각 sample의 내부 weight 평균을 1로 고정
# - 전역 physics coefficient `0.1/240`은 fixed/adaptive에서 동일
# - `MATERIAL_PRIOR_WEIGHT=0.0`이 기본값: 불연속 $k$에서 strong-form 미분 오차를 곧바로 강화하지 않음

# %%
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
    # 전체 240x240 element 수로 나누므로 fixed mode는 원본 F.l1_loss와 같습니다.
    pde_loss = (weights * residual_padded.abs()).sum() / residual_padded.numel()
    total_loss = data_loss + OFFICIAL_DX * OFFICIAL_PHYSICS_WEIGHT * pde_loss
    return {
        "prediction": prediction,
        "residual": residual_padded,
        "weights": weights,
        "data": data_loss,
        "pde": pde_loss,
        "total": total_loss,
    }


print("mode / steps / validation:", QUICK_MODE, STEPS_PER_VARIANT, VALIDATION_LIMIT)

# %% [markdown]
# ### Colab smoke test — forward, backward, 불변조건
#
# 긴 학습 전에 다음을 확인합니다.
#
# 1. fixed loss가 기존 공식 loss와 수치적으로 같은가?
# 2. adaptive weight가 `[B,1,240,240]`이고 내부 평균이 1인가?
# 3. weight에 gradient가 연결되지 않았는가?
# 4. adaptive total loss가 FNO 파라미터까지 정상적으로 backward되는가?

# %%
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
print("original/fixed PDE:", float(original_check["pde"]), float(fixed_check["pde"]))
print("adaptive weight mean:", float(interior_weight_mean))
print("adaptive weight min/max:",
      float(adaptive_check["weights"][adaptive_check["weights"] > 0].min()),
      float(adaptive_check["weights"].max()))

assert torch.allclose(original_check["pde"], fixed_check["pde"], atol=1e-6)
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

# %% [markdown]
# ## 5. 같은 초기값·같은 순서로 fixed와 adaptive 학습
#
# QUICK mode의 목적은 실행 검증입니다. 12 step 결과로 우열을 주장하지 않습니다.
# 실제 비교 시 `QUICK_MODE=False`로 바꾸고 여러 seed를 반복해야 합니다.

# %%
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
    history = {"data": [], "pde": [], "total": [], "lr": []}
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

        for key in ["data", "pde", "total"]:
            history[key].append(float(losses[key].detach()))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if step == 1 or step % max(1, STEPS_PER_VARIANT // 4) == 0:
            print(
                f"{mode:8s} {step:4d}/{STEPS_PER_VARIANT} | "
                f"data={history['data'][-1]:.4e} | "
                f"pde={history['pde'][-1]:.4e}"
            )

    print(mode, "elapsed:", f"{time.perf_counter() - started:.1f}s")
    return model, history


fixed_model, fixed_history = train_variant("fixed")
adaptive_model, adaptive_history = train_variant("adaptive")

# %% [markdown]
# ## 6. Validation: 전체 오차와 permeability 급변 영역을 분리
#
# 전체 relative L2와 PDE L1 외에 $|\nabla k|$ 상위 10% 영역의 residual을 따로 측정합니다.
# 재료 급변도는 여기서 **평가 마스크**로만 사용하며 기본 adaptive loss에는 넣지 않습니다.

# %%
# 6-1. 동일 validation subset 평가와 시각화
@torch.no_grad()
def evaluate_variant(model, mode):
    model.eval()
    metrics = {"mse": [], "relative_l2": [], "pde_l1": [], "interface_pde_l1": []}
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

        metrics["mse"].append(float(F.mse_loss(prediction, outvar)))
        metrics["relative_l2"].append(float(
            torch.linalg.vector_norm(prediction - outvar)
            / torch.linalg.vector_norm(outvar).clamp_min(1e-12)
        ))
        metrics["pde_l1"].append(float(residual_abs.mean()))
        metrics["interface_pde_l1"].append(float(
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
print("fixed   :", fixed_metrics)
print("adaptive:", adaptive_metrics)

fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
for row, (name, shown) in enumerate([
    ("fixed", fixed_shown), ("adaptive", adaptive_shown)
]):
    panels = [
        (shown["k"][0, 0], "k", "viridis"),
        (shown["target"][0, 0], "target u", "magma"),
        (shown["prediction"][0, 0], f"{name} u", "magma"),
        (shown["residual"][0, 0], f"{name} |residual|", "inferno"),
        (shown["weights"][0, 0], f"{name} weight", "plasma"),
    ]
    for axis, (image, title, cmap) in zip(axes[row], panels):
        rendered = axis.imshow(image.numpy(), origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xticks([]); axis.set_yticks([])
        plt.colorbar(rendered, ax=axis, fraction=0.046)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
axes[0].semilogy(fixed_history["total"], label="fixed")
axes[0].semilogy(adaptive_history["total"], label="adaptive")
axes[0].set_title("total loss"); axes[0].legend(); axes[0].grid(alpha=0.25)
axes[1].semilogy(fixed_history["pde"], label="fixed")
axes[1].semilogy(adaptive_history["pde"], label="adaptive")
axes[1].set_title("weighted PDE L1"); axes[1].legend(); axes[1].grid(alpha=0.25)
plt.show()

# %% [markdown]
# ## 해석 기준과 다음 ablation
#
# QUICK mode가 통과해도 adaptive 방법이 더 좋다고 결론 내리면 안 됩니다. 다음 순서로 실험을 늘립니다.
#
# 1. 3개 이상 seed에서 fixed/adaptive 반복
# 2. validation relative L2, 전체 PDE L1, interface PDE L1의 평균·표준편차 보고
# 3. `ADAPTIVE_ALPHA = 1, 2, 4` 비교
# 4. 그 뒤에만 `MATERIAL_PRIOR_WEIGHT > 0` 실험
# 5. material prior를 켤 때는 flux-conservative residual과 PhysicsInformer residual을 별도 비교

# %%
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

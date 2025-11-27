# syntax=docker/dockerfile:1.4-labs

########################################
# 1) libriichi 빌드 스테이지 (Rust)
########################################
FROM rust:bookworm AS libriichi_build

# 빌드 도구
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config python3 git && \
    rm -rf /var/lib/apt/lists/*

    # crates.io sparse 가속(선택)
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
ENV PYO3_USE_ABI3=1

# ✅ nightly 설치 후 기본 툴체인으로 설정
RUN rustup toolchain install nightly && rustup default nightly && rustc --version && cargo --version

WORKDIR /mortal

# 워크스페이스 메타 먼저 복사 (캐시 최적화)
COPY Cargo.toml Cargo.lock ./
# 라이브러리/워크스페이스 멤버 복사
COPY libriichi ./libriichi
COPY exe-wrapper ./exe-wrapper
# 필요하면 다른 멤버도 추가
# COPY some-other-crate ./some-other-crate

# 캐시 마운트 + ✅ nightly로 빌드
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo +nightly build -p libriichi --lib --release

########################################
# 2) 런타임 스테이지 (CUDA + Python)
########################################
# CUDA 12.1 런타임: PyTorch cu121 휠과 호환성이 가장 좋음
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python + 기본 유틸
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-distutils ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# PyTorch (CUDA 12.1 빌드) + 기타 파이썬 의존성
# ※ cu121 빌드는 CUDA 런타임을 동봉하므로 추가 툴킷 설치 불필요
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300

# RUN apt-get update && apt-get install -y --no-install-recommends coreutils && rm -rf /var/lib/apt/lists/*

# 1) 휠만 먼저 다운로드(재시도). 네트워크 문제는 여기에서만 발생.
RUN --mount=type=cache,target=/root/.cache/pip \
    python - <<'PY'
import subprocess, sys, time, os, pathlib
wdir = pathlib.Path("/tmp/wheels"); wdir.mkdir(parents=True, exist_ok=True)
pkgs = [
  "torch==2.4.1+cu121",
  "torchvision==0.19.1+cu121",
  "torchaudio==2.4.1+cu121",
]
args = [sys.executable, "-m", "pip", "download", "--dest", str(wdir),
        "--extra-index-url", "https://download.pytorch.org/whl/cu121", *pkgs]
for i in range(5):
    print(f"[pip download try {i+1}/5]", flush=True)
    rc = subprocess.call(args)
    if rc == 0:
        break
    time.sleep(15)
else:
    sys.exit(1)
PY

# 2) 오프라인 설치(네트워크 의존 제거). 이 레이어는 보통 빨리/안정적으로 완료됨.
RUN python -m pip install --upgrade pip && \
    pip install /tmp/wheels/* && \
    pip install toml tqdm tensorboard && \
    rm -rf /tmp/wheels

# 워크디렉토리
WORKDIR /mortal

# 프로젝트 파이썬 코드 복사
COPY mortal/ ./

# numpy._core 경고 회피용 safe_globals 등록(기존과 동일 패치)
RUN sed -i "1i from torch.serialization import add_safe_globals\nimport numpy as np\nadd_safe_globals([np.core.multiarray.scalar])" /mortal/mortal.py

# libriichi 공유 라이브러리 복사(이름이 liblibriichi.so 혹은 변형일 수 있어 와일드카드)
COPY --from=libriichi_build /mortal/target/release/*libriichi*.so ./libriichi.so

# 기본 설정
ENV MORTAL_CFG=/mortal/config.toml \
    # 스레드/IPC 권장 (실행시 --ipc=host, --shm-size도 함께 설정 권장)
    OMP_NUM_THREADS=6 \
    MKL_NUM_THREADS=6 \
    PYTORCH_NUM_THREADS=6

# 기본 config.toml (필요시 실행 시 -v로 교체 가능)
COPY <<'EOF' /mortal/config.toml
[control]
state_file = '/mnt/mortal.pth'
[resnet]
conv_channels = 192
num_blocks = 40
enable_bn = true
bn_momentum = 0.99
EOF

# 모델/체크포인트 마운트
VOLUME /mnt

# 실행 엔트리포인트
# GPU 사용하려면: docker run --gpus all --ipc=host --shm-size=2g -v D:\mortal\models:/mnt <image> 2
ENTRYPOINT ["python", "mortal.py"]

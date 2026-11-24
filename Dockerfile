# Образ бенчмарка: TensorRT + PyTorch + PyNvVideoCodec + ffmpeg.
# Версия базового образа фиксирует CUDA и TensorRT; переопределяется через --build-arg.
ARG TRT_IMAGE=nvcr.io/nvidia/tensorrt:25.01-py3
FROM ${TRT_IMAGE}
ARG TRT_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    # video нужен для доступа к NVDEC внутри контейнера
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    BENCH_IMAGE=${TRT_IMAGE} \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg util-linux curl unzip \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml ./
# tensorrt берётся из базового образа, поэтому extra "trt" не ставим
RUN uv pip compile pyproject.toml --extra gpu --extra export --extra dev -o /tmp/requirements.txt \
 && uv pip install --system --break-system-packages -r /tmp/requirements.txt

COPY . .

CMD ["python", "scripts/env_info.py"]

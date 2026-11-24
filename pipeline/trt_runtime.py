"""Обёртка над TensorRT engine с динамическим батчем; вход и выходы — torch-тензоры на GPU."""
from __future__ import annotations

import torch
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)

_DTYPES = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}
if hasattr(trt, "int64"):
    _DTYPES[trt.int64] = torch.int64
if hasattr(trt, "bfloat16"):
    _DTYPES[trt.bfloat16] = torch.bfloat16


class TrtModel:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f:
            data = f.read()
        self.path = engine_path
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"не удалось загрузить engine {engine_path} (другая версия TensorRT?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(self.inputs) != 1:
            raise RuntimeError(f"ожидается один вход, получено {self.inputs}")
        self.input_name = self.inputs[0]
        _, _, mx = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.max_batch = int(mx[0])
        self.input_shape = tuple(int(d) for d in mx[1:])
        self.input_dtype = _DTYPES[self.engine.get_tensor_dtype(self.input_name)]

    def __call__(self, x: torch.Tensor, stream: torch.cuda.Stream | None = None) -> list[torch.Tensor]:
        stream = stream or torch.cuda.current_stream()
        parts = [self._run(chunk, stream) for chunk in x.split(self.max_batch)]
        if len(parts) == 1:
            return parts[0]
        return [torch.cat(p) for p in zip(*parts)]

    def _run(self, x: torch.Tensor, stream) -> list[torch.Tensor]:
        x = x.to(self.input_dtype).contiguous()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        outs = []
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            t = torch.empty(shape, dtype=_DTYPES[self.engine.get_tensor_dtype(name)], device=x.device)
            self.context.set_tensor_address(name, t.data_ptr())
            outs.append(t)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"ошибка выполнения {self.path}")
        # x должен жить до конца выполнения на потоке: вызывающий синхронизирует поток
        self._keepalive = x
        return outs

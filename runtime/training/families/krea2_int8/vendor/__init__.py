"""Vendored Krea2 INT8 (ConvRot) 依赖模块。

直接复制自 kohya-ss/musubi-tuner（Apache-2.0），仅做了 import 路径改写以便
在本仓库内自包含运行。上游固定提交见 THIRD_PARTY_NOTICES.md。

- ``convrot_int8_kernels.py``：ConvRot INT8 核（Hadamard 旋转 + Triton GEMM，
  无 triton 时退化为 eager dequant 路径），自包含。
- ``convrot_int8_utils.py``：ConvRot INT8 量化器 + ``nn.Linear`` 前向 monkey-patch。
- ``safetensors_utils.py`` / ``device_utils.py``：前者依赖的流式读取/内存工具。

本包内容按原样复制（少改动），业务封装在 ``krea2_int8.quant_int8`` /
``krea2_int8.loader`` 中，勿在本包内写业务逻辑。
"""

# 复制来源版权信息（Apache-2.0）：
# Copyright 2026 Kohya S. and musubi-tuner contributors.

from functools import cache
from typing import Any

import torch

SUPPORT_QUANT = False
try:
    from bitsandbytes.nn import LinearNF4, Linear8bitLt, LinearFP4

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class LinearNF4(nn.Linear):
        pass

    class Linear8bitLt(nn.Linear):
        pass

    class LinearFP4(nn.Linear):
        pass


try:
    from quanto.nn import QLinear, QConv2d, QLayerNorm

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class QLinear(nn.Linear):
        pass

    class QConv2d(nn.Conv2d):
        pass

    class QLayerNorm(nn.LayerNorm):
        pass


try:
    from optimum.quanto.nn import (
        QLinear as QLinearOpt,
        QConv2d as QConv2dOpt,
        QLayerNorm as QLayerNormOpt,
    )

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class QLinearOpt(nn.Linear):
        pass

    class QConv2dOpt(nn.Conv2d):
        pass

    class QLayerNormOpt(nn.LayerNorm):
        pass


try:
    from torchao.utils import TorchAOBaseTensor
except ImportError:
    TorchAOBaseTensor = None

from ..logging import logger

QuantLinears = (
    Linear8bitLt,
    LinearFP4,
    LinearNF4,
    QLinear,
    QConv2d,
    QLayerNorm,
    QLinearOpt,
    QConv2dOpt,
    QLayerNormOpt,
)


def get_qdata(module) -> Any | None:
    """Quantized weight data of module. Type(T) is NOT torch.Tensor"""
    weight_data = getattr(getattr(module, "weight", None), "data", None)
    if type(weight_data) is not torch.Tensor:
        return weight_data


def is_quantized_weight(module) -> bool:
    """Tensor-subclass quantization (torchao, bnb, ...): the module stays a
    plain Linear/Conv, only weight.data is a subclass like NVFP4Tensor.
    Weight-space math on such tensors raises NotImplementedError."""
    return get_qdata(module) is not None


def has_torchao_quantized_weight(module) -> bool:
    """Whether the weight is a torchao tensor with a real dequantize().
    torchao monkeypatches Tensor.dequantize, so duck-typing is unsafe:
    on a plain tensor it upcasts to fp32, on bnb Params4bit it ignores
    quant_state and returns garbage."""
    return TorchAOBaseTensor is not None and isinstance(
        get_qdata(module), TorchAOBaseTensor
    )


@cache
def log_bypass():
    return logger.warning(
        "Using bnb/quanto/optimum-quanto with LyCORIS will enable force-bypass mode."
    )


@cache
def log_fp8_bypass():
    return logger.warning(
        "Using weight-only FP8 Linear with LyCORIS will enable force-bypass mode."
    )


@cache
def log_suspect():
    return logger.warning(
        "Non-native Linear detected but bypass_mode is not set. "
        "Automatically using force-bypass mode to avoid possible issues. "
        "Please set bypass_mode=False explicitly if there are no quantized layers."
    )

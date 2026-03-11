from .params import flatten_params, param_count, unflatten_like
from . import resnet_cifar as _resnet_cifar
from .small_resnet_cifar_ln import create_small_resnet_ln


def create_model(
    width_multiplier: float = 1.0,
    num_classes: int = 10,
    arch: str = "resnet18",
    **kwargs,
):
    """
    Model factory with architecture switch.

    - arch="resnet18": standard ResNet-18 for CIFAR with BatchNorm.
    - arch="small_resnet_ln": small CIFAR ResNet with LayerNorm.
    """
    if arch == "resnet18":
        return _resnet_cifar.create_model(
            width_multiplier=width_multiplier,
            num_classes=num_classes,
        )
    if arch == "small_resnet_ln":
        num_blocks = int(kwargs.pop("num_blocks", 2))
        return create_small_resnet_ln(
            width_multiplier=width_multiplier,
            num_classes=num_classes,
            num_blocks=num_blocks,
        )
    raise ValueError(f"Unknown architecture: {arch}")


__all__ = [
    "create_model",
    "flatten_params",
    "unflatten_like",
    "param_count",
]

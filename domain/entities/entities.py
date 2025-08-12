from numpy._typing._shape import _Shape

from typing import TypeAlias

from dataclasses import dataclass

from numpy import uint8


@dataclass(frozen=True)
class Image:
    size: tuple[uint8, uint8]
    shape: _Shape
    dtype: TypeAlias = uint8


@dataclass(frozen=True)
class FinalClass:
    shape: _Shape
    dtype: object

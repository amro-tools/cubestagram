from typing import Protocol

import numpy.typing as npt

from dataclasses import dataclass


@dataclass
class FrameInfo:
    positions: npt.ArrayLike
    box_lengths: tuple[float]

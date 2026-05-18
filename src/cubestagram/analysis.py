from cubestagram.scaffold import FrameInfo
from typing import Generator
import numpy.typing as npt
import numba


def analyze(frame_provider: Generator[FrameInfo]):
    for frame_info in frame_provider:
        ...

from typing import Protocol

import numpy.typing as npt
from numba import njit
from dataclasses import dataclass
import numpy as np


@dataclass
class FrameInfo:
    positions: npt.NDArray
    box_lengths: tuple[float]
    masses: npt.NDArray  # mass of each atom (assume sorted positions)


@njit
def partition_pos_into_cells(
    pos_frame: npt.NDArray,
    nx: int,
    ny: int,
    nz: int,
    box_lengths: list[float],
    masses: npt.NDArray,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:

    assert len(box_lengths) == 3

    output_number_frame = np.zeros((nx, ny, nz), dtype=np.int32)
    output_density_frame = np.zeros((nx, ny, nz), dtype=np.float64)

    # Cell lengths
    cell_lx = box_lengths[0] / nx
    cell_ly = box_lengths[1] / ny
    cell_lz = box_lengths[2] / nz
    vol = cell_lx * cell_ly * cell_lz

    # each atom
    for i_pos, i_mass in zip(pos_frame, masses):
        # Wrap back into box if needed
        i_pos[0] = i_pos[0] % box_lengths[0]
        i_pos[1] = i_pos[1] % box_lengths[1]
        i_pos[2] = i_pos[2] % box_lengths[2]

        cell_index_x = int(i_pos[0] / cell_lx)
        cell_index_y = int(i_pos[1] / cell_ly)
        cell_index_z = int(i_pos[2] / cell_lz)

        output_number_frame[cell_index_x, cell_index_y, cell_index_z] += 1
        output_density_frame[cell_index_x, cell_index_y, cell_index_z] += i_mass / vol

    return output_density_frame, output_number_frame

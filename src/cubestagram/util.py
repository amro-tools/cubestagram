from cubestagram.analysis import Config
from numba import njit

def decide_number_of_cells(
    config: Config, box_lengths: tuple[float]
) -> tuple[int, int, int]:
    assert len(box_lengths) == 3
    nx = round(box_lengths[0] / config.target_lx)
    ny = round(box_lengths[1] / config.target_ly)
    nz = round(box_lengths[2] / config.target_lz)

    return (nx, ny, nz)

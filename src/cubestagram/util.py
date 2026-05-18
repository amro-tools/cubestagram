from numba import njit


def decide_number_of_cells(
    target_lx: float, target_ly: float, target_lz: float, box_lengths: tuple[float]
) -> tuple[int, int, int]:
    assert len(box_lengths) == 3
    nx = max(round(box_lengths[0] / target_lx), 1)
    ny = max(round(box_lengths[1] / target_ly), 1)
    nz = max(round(box_lengths[2] / target_lz), 1)

    return (nx, ny, nz)

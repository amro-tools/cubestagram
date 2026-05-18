from cubestagram.scaffold import FrameInfo, partition_pos_into_cells
from typing import Generator
import numpy.typing as npt
from dataclasses import dataclass
from cubestagram.util import decide_number_of_cells


@dataclass
class Config:
    target_lx: float
    target_ly: float
    target_lz: float


@dataclass
class AnalysisState:
    nx: int | None = None
    ny: int | None = None
    nz: int | None = None
    density_over_traj: list[npt.ArrayLike] | None = None


def analyze(frame_provider: Generator[FrameInfo], config: Config):

    # Initialize analysis state
    state = AnalysisState()

    for frame_info in frame_provider:
        positions = frame_info.positions
        box_lengths = frame_info.box_lengths

        # Update state if the number of cells has not been set
        if any(n is None for n in [state.nx, state.ny, state.nz]):
            state.nx, state.ny, state.nz = decide_number_of_cells(
                config=config, box_lengths=box_lengths
            )
            # Initialize the output list
            state.density_over_traj = []

        assert state.density_over_traj is not None
        assert state.nx is not None
        assert state.ny is not None
        assert state.nz is not None

        # Get the number density
        density_per_frame = partition_pos_into_cells(
            pos_frame=positions,
            nx=state.nx,
            ny=state.ny,
            nz=state.nz,
            box_lengths=list(box_lengths),
        )

        state.density_over_traj.append(density_per_frame)

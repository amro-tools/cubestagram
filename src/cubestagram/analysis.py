from cubestagram.scaffold import FrameInfo, partition_pos_into_cells
from typing import Generator
import numpy.typing as npt
from dataclasses import dataclass
from cubestagram.util import decide_number_of_cells
from pathlib import Path
import numpy as np


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

    def to_file(self, filename: Path):

        n_frames = (
            len(self.density_over_traj) if self.density_over_traj is not None else 0
        )

        kwargs = {"nx": self.nx, "ny": self.ny, "nz": self.nz, "n_frames": n_frames}

        if self.density_over_traj is not None:
            kwargs.update(
                {
                    f"density_over_traj_{i}": dens
                    for i, dens in enumerate(self.density_over_traj)
                }
            )

        np.savez_compressed(filename, **kwargs)

    @staticmethod
    def from_file(filename: Path):
        data = np.load(filename)

        if data["n_frames"] == 0:
            density_over_traj = None
        else:
            density_over_traj = [
                data[f"density_over_traj_{i}"] for i in range(data["n_frames"])
            ]

        return AnalysisState(
            nx=data["nx"],
            ny=data["ny"],
            nz=data["nz"],
            density_over_traj=density_over_traj,
        )


def analyze(frame_provider: Generator[FrameInfo], config: Config):

    # Initialize analysis state
    state = AnalysisState()

    for frame_info in frame_provider:
        positions = frame_info.positions
        box_lengths = frame_info.box_lengths

        # Update state if the number of cells has not been set
        if any(n is None for n in [state.nx, state.ny, state.nz]):
            state.nx, state.ny, state.nz = decide_number_of_cells(
                target_lx=config.target_lx,
                target_ly=config.target_ly,
                target_lz=config.target_lz,
                box_lengths=box_lengths,
            )
            # Initialize the output list
            state.density_over_traj = []

        assert state.density_over_traj is not None
        assert state.nx is not None
        assert state.ny is not None
        assert state.nz is not None

        print(f"{state.nx=}")
        print(state.ny)
        print(state.nz)

        # Get the number density
        density_frame = partition_pos_into_cells(
            pos_frame=positions,
            nx=state.nx,
            ny=state.ny,
            nz=state.nz,
            box_lengths=list(box_lengths),
        )

        state.density_over_traj.append(density_frame)

    return state

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
    density_over_traj: list[npt.ArrayLike] | None = None  # mass density
    number_over_traj: list[npt.ArrayLike] | None = None  # number density
    n_frames: int = 0

    def to_file(self, filename: Path):

        kwargs = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "n_frames": self.n_frames,
        }

        if self.density_over_traj is not None:
            kwargs.update(
                {
                    f"density_over_traj_{i}": dens
                    for i, dens in enumerate(self.density_over_traj)
                }
            )

        if self.number_over_traj is not None:
            kwargs.update(
                {
                    f"number_over_traj_{i}": dens
                    for i, dens in enumerate(self.number_over_traj)
                }
            )

        np.savez_compressed(filename, **kwargs)

    @staticmethod
    def from_file(filename: Path):
        data = np.load(filename)

        if data["n_frames"] == 0:
            density_over_traj = None
            number_over_traj = None
        else:
            density_over_traj = [
                data[f"density_over_traj_{i}"] for i in range(data["n_frames"])
            ]
            number_over_traj = [
                data[f"number_over_traj_{i}"] for i in range(data["n_frames"])
            ]

        return AnalysisState(
            nx=data["nx"],
            ny=data["ny"],
            nz=data["nz"],
            density_over_traj=density_over_traj,
            number_over_traj=number_over_traj,
            n_frames=data["n_frames"],
        )


def analyze(frame_provider: Generator[FrameInfo], config: Config):

    # Initialize analysis state
    state = AnalysisState()

    for frame_info in frame_provider:
        positions = frame_info.positions
        box_lengths = frame_info.box_lengths
        state.n_frames += 1

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
            state.number_over_traj = []

        assert state.density_over_traj is not None
        assert state.number_over_traj is not None
        assert state.nx is not None
        assert state.ny is not None
        assert state.nz is not None

        # Get the number density and mass density
        density_frame, number_frame = partition_pos_into_cells(
            pos_frame=positions,
            nx=state.nx,
            ny=state.ny,
            nz=state.nz,
            box_lengths=list(box_lengths),
        )

        state.density_over_traj.append(density_frame)
        state.number_over_traj.append(number_frame)

    return state

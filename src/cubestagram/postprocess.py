import numpy as np
import numpy.typing as npt
from cubestagram.analysis import AnalysisState
from numba import njit


def density_to_histogram_frame(
    density_frame: npt.ArrayLike, nbins: int = 10
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    return np.histogram(density_frame, bins=nbins)


@njit
def loop_over_counts(flat_number_densities):
    max_count = np.max(flat_number_densities)
    counts = np.zeros(shape=(max_count + 1))
    for n in flat_number_densities:
        counts[n] += 1
    return counts 


def analyze_cube_counts(state: AnalysisState):
    # Flatten the array of number densities
    assert state.number_over_traj is not None
    flat_number_densities = np.ravel(np.array(state.number_over_traj, dtype=np.int32))

    counts = loop_over_counts(flat_number_densities)
    return counts

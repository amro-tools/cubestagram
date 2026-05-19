import numpy as np
import numpy.typing as npt
from cubestagram.analysis import AnalysisState
from numba import njit
from scipy.special import factorial
from scipy.optimize import curve_fit


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


def analyze_cube_counts(state: AnalysisState) -> npt.ArrayLike:
    # Flatten the array of number densities
    assert state.number_over_traj is not None
    flat_number_densities = np.ravel(np.array(state.number_over_traj, dtype=np.int32))

    cube_counts = loop_over_counts(flat_number_densities)
    return cube_counts / state.n_frames


def fitting_poisson_to_dilute(
    cube_counts: npt.NDArray,
    cutoff: int,
    A0: float | None = None,
    lambda0: float | None = None,
):

    def fit_func(bead_count, A, lam):
        return A * lam**bead_count * np.exp(-lam) / factorial(bead_count)

    bead_counts = np.arange(0, cutoff, step=1, dtype=np.int32)  # only bead
    cube_counts = cube_counts[:cutoff]

    assert len(bead_counts) == len(cube_counts)

    if A0 is None:
        A0 = np.sum(cube_counts)

    if lambda0 is None:
        lambda0 = 1e-2

    p0 = [A0, lambda0]
    popt, pcov = curve_fit(fit_func, xdata=bead_counts, ydata=cube_counts, p0=p0)

    return popt, p0, fit_func

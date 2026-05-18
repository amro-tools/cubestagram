import numpy as np
import numpy.typing as npt


def density_to_histogram_frame(
    density_frame: npt.ArrayLike, nbins: int = 10
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    return np.histogram(density_frame, bins=nbins)

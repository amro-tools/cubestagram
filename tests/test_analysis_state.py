from cubestagram.analysis import AnalysisState
import numpy as np


def test_analysis_state():
    nx = 5
    ny = 1
    nz = 2
    test_density = [np.random.uniform(size=(nx, ny, nz))]

    state = AnalysisState(nx, ny, nz, density_over_traj=test_density)

    state.to_file("bla.npz")

    state2 = AnalysisState.from_file("bla.npz")

    for d1, d2 in zip(state.density_over_traj, state2.density_over_traj):
        assert np.all(d1 == d2)


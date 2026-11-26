"""Headless smoke test for the experiment window (no display required).

Instantiates the full matplotlib widget layout under the Agg backend,
fires recompute events for every control combination, and exercises the
failure-feedback path with an out-of-range DoP.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polrender.gui import ExperimentWindow
from polrender.stokes import UnphysicalStokesError


def main():
    win = ExperimentWindow()
    assert win.last_result is not None, "initial computation did not run"
    assert win.last_result.valid.sum() > 0

    # Sweep every slider through representative values.
    for sl in (win.sl_psi, win.sl_dop, win.sl_chi, win.sl_n2,
               win.sl_ana, win.sl_ext):
        for v in (sl.valmin, (sl.valmin + sl.valmax) / 2, sl.valmax):
            sl.set_val(v)
        assert win.last_result is not None

    # Switch scene and degeneracy policy.
    win.rad_scene.set_active(1)
    assert win.last_result.valid.sum() > 0
    win.rad_policy.set_active(1)
    assert win.last_result is not None
    win.rad_scene.set_active(0)

    # Unpolarized source through an ideal analyzer: after dielectric
    # reflection the light is partially s-polarized, so DoP must be
    # positive and the crossed analyzer pair sums to the reflected I.
    win.sl_ext.set_val(0.0)
    win.sl_dop.set_val(0.0)
    res = win.last_result
    assert float(res.dop_map()[res.valid].min()) > 0.0
    i0 = res.intensity[0][res.valid]
    i90 = res.intensity[2][res.valid]
    total = res.stokes_grid[res.valid][:, 0]
    import numpy as np
    assert np.allclose(i0 + i90, total, atol=1e-12)
    # Before any interface, an ideal polarizer gives I/2 at every angle.
    from polrender import from_components, analyzer_intensity
    sun = from_components(1.0, dop=0.0)
    assert all(abs(analyzer_intensity(sun, a) - 0.5) < 1e-12
               for a in (0.0, 30.0, 90.0, 140.0))

    # Non-ideal analyzer: extinction floor keeps the crossed channel dark
    # but non-zero, and every output state stays physical.
    win.sl_dop.set_val(1.0)
    win.sl_ext.set_val(0.05)
    res = win.last_result
    crossed = res.intensity[2][res.valid]
    assert float(crossed.min()) > 0.0
    from polrender import check_feasible
    for j, i in np.argwhere(res.valid)[::997]:
        assert check_feasible(res.stokes_grid[j, i]).ok
    win.sl_ext.set_val(0.0)

    # Failure feedback: force an infeasible DoP through the build path.
    win.sl_dop.set_val(1.0)
    original = win._build_source
    def bad_source():
        raise UnphysicalStokesError("simulated DoP=1.5 rejection")
    win._build_source = bad_source
    win.run(None)
    status = win.text_status.get_text()
    assert "rejected" in status and "DoP=1.5" in status, status
    win._build_source = original
    win.run(None)
    assert win.text_status.get_text().startswith("OK")

    plt.close(win.fig)
    print("GUI smoke test passed: widget layout, kernel updates, "
          "unpolarized symmetry and failure feedback all exercised.")


if __name__ == "__main__":
    main()

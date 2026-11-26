"""Interactive polarization light-experiment window.

The window is driven entirely by the Stokes/Mueller kernel in :mod:`polrender`:
sliders set physical source/analyzer parameters, the renderer propagates full
Stokes states through one or two interface reflections, and every displayed
image or number comes from a kernel call that re-checks the physical cone.
Invalid control values (DoP outside [0,1], bad index, ...) are caught at the
kernel boundary and reported back in the diagnostics panel.

Launch:
    .venv/bin/python -m polrender.gui
"""

from __future__ import annotations

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
import numpy as np

from .mueller import analyzer_intensity, brewster_angle
from .path import PathEngine
from .scene import CorrugatedPlane, VGroove, PolarizedRenderer
from .stokes import (
    UnphysicalStokesError,
    check_feasible,
    degrees_of_polarization,
    from_components,
    linear,
    zero,
)

ANALYZER_ANGLES = (0.0, 45.0, 90.0, 135.0)


class ExperimentWindow:
    def __init__(self):
        self.renderer_transport = PolarizedRenderer("transport")
        self.renderer_reset = PolarizedRenderer("reset")
        self.last_result = None
        self.fig = plt.figure(figsize=(14.5, 9.0))
        self.fig.canvas.manager.set_window_title(
            "Polarization propagation experiment")
        gs = self.fig.add_gridspec(
            3, 4, left=0.055, right=0.985, top=0.94, bottom=0.30,
            hspace=0.55, wspace=0.28,
        )
        self.ax_img = [
            self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[0, 2]), self.fig.add_subplot(gs[0, 3]),
        ]
        self.ax_dop = self.fig.add_subplot(gs[1, 0])
        self.ax_i = self.fig.add_subplot(gs[1, 1])
        self.ax_sweep = self.fig.add_subplot(gs[1, 2])
        self.ax_text = self.fig.add_subplot(gs[1, 3])
        self.ax_text.axis("off")

        # Control axes.
        ctl = dict(facecolor="#eef0f3")
        self.ax_s_psi = self.fig.add_axes([0.07, 0.235, 0.26, 0.025], **ctl)
        self.ax_s_dop = self.fig.add_axes([0.07, 0.195, 0.26, 0.025], **ctl)
        self.ax_s_chi = self.fig.add_axes([0.07, 0.155, 0.26, 0.025], **ctl)
        self.ax_s_n2 = self.fig.add_axes([0.42, 0.235, 0.26, 0.025], **ctl)
        self.ax_s_ana = self.fig.add_axes([0.42, 0.195, 0.26, 0.025], **ctl)
        self.ax_s_ext = self.fig.add_axes([0.42, 0.155, 0.26, 0.025], **ctl)
        self.ax_scene = self.fig.add_axes([0.74, 0.135, 0.22, 0.145], **ctl)
        self.ax_policy = self.fig.add_axes([0.74, 0.045, 0.10, 0.075], **ctl)
        self.ax_run = self.fig.add_axes([0.86, 0.055, 0.10, 0.055])
        self.ax_status = self.fig.add_axes([0.07, 0.045, 0.62, 0.075])
        self.ax_status.axis("off")

        self.sl_psi = Slider(self.ax_s_psi, "source psi [deg]", 0.0, 180.0,
                             valinit=0.0, valstep=1.0)
        self.sl_dop = Slider(self.ax_s_dop, "source DoP", 0.0, 1.0,
                             valinit=1.0, valstep=0.01)
        self.sl_chi = Slider(self.ax_s_chi, "ellipticity chi [deg]", -45.0,
                             45.0, valinit=0.0, valstep=1.0)
        self.sl_n2 = Slider(self.ax_s_n2, "index n2", 1.05, 3.0,
                            valinit=1.5, valstep=0.01)
        self.sl_ana = Slider(self.ax_s_ana, "analyzer axis [deg]", 0.0, 180.0,
                             valinit=0.0, valstep=1.0)
        self.sl_ext = Slider(self.ax_s_ext, "analyzer extinction", 0.0, 0.2,
                             valinit=0.0, valstep=0.005)
        self.rad_scene = RadioButtons(
            self.ax_scene, ("corrugated plane (1 bounce)",
                            "V-groove (2 bounces)"), active=0)
        self.rad_policy = RadioButtons(self.ax_policy,
                                       ("transport", "reset"), active=0)
        self.btn_run = Button(self.ax_run, "Compute /\nrecompute",
                              color="#9ecb9b", hovercolor="#c9e6c5")
        self.btn_run.on_clicked(self.run)
        self.rad_scene.on_clicked(lambda _: self.run(None))
        self.rad_policy.on_clicked(lambda _: self.run(None))
        # Slider live updates are cheap; still routed through the kernel.
        for sl in (self.sl_psi, self.sl_dop, self.sl_chi, self.sl_n2,
                   self.sl_ana, self.sl_ext):
            sl.on_changed(lambda _: self.run(None))

        self.text_status = self.ax_status.text(
            0.0, 0.5, "", va="center", family="monospace", fontsize=9,
            color="#7a1f1f")
        self.fig.suptitle(
            "Stokes/Mueller polarized reflection -- virtual rotating "
            "analyzer in front of the sensor", fontsize=13)
        self.run(None)

    # ------------------------------------------------------------------

    def _build_source(self):
        """Source Stokes state; invalid values are rejected at this boundary."""
        dop = float(self.sl_dop.val)
        if not (0.0 <= dop <= 1.0):
            raise UnphysicalStokesError(f"source DoP {dop} outside [0,1]")
        if dop == 0.0:
            return from_components(1.0, dop=0.0)
        return from_components(1.0, dop=dop,
                               psi_deg=float(self.sl_psi.val),
                               chi_deg=float(self.sl_chi.val))

    def run(self, _event):
        try:
            source = self._build_source()
            check_feasible(source, raise_on_fail=True)
            n2 = complex(float(self.sl_n2.val))
            renderer = (self.renderer_transport
                        if self.rad_policy.value_selected.startswith("transport")
                        else self.renderer_reset)
            if self.rad_scene.value_selected.startswith("corrugated"):
                scene = CorrugatedPlane(amplitude=0.35, period=2.0, n2=n2)
                result = renderer.render_corrugated_plane(
                    scene, source, width=220, height=120, extent=1.0,
                    analyzer_angles=ANALYZER_ANGLES,
                    extinction=float(self.sl_ext.val))
            else:
                scene = VGroove(half_angle_deg=45.0, depth=1.0, n2=n2)
                result = renderer.render_v_groove(
                    scene, source, width=220, height=120,
                    analyzer_angles=ANALYZER_ANGLES,
                    extinction=float(self.sl_ext.val))
            self.last_result = result
            self._draw(result, source, n2)
            msg = "OK: kernel propagation succeeded"
            if result.diagnostics:
                msg += " | warnings: " + " ; ".join(result.diagnostics[:3])
            self._set_status(msg, ok=not result.diagnostics)
        except (UnphysicalStokesError, ValueError) as exc:
            # Failure is fed back into the polarization diagnostics area;
            # no unphysical state ever reaches the color conversion.
            self._set_status(f"kernel rejected input: {exc}", ok=False)
        except Exception as exc:  # pragma: no cover - UI safety net
            self._set_status(f"unexpected error: {type(exc).__name__}: {exc}",
                             ok=False)

    # ------------------------------------------------------------------

    def _draw(self, result, source, n2):
        for ax in self.ax_img:
            ax.clear()
        vmax = max(result.stokes_grid[..., 0].max(), 1e-9)
        for k, ang in enumerate(ANALYZER_ANGLES):
            ax = self.ax_img[k]
            im = ax.imshow(result.intensity[k], cmap="gray",
                           vmin=0.0, vmax=vmax, aspect="auto")
            ax.set_title(f"analyzer {ang:g} deg", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # DoP and total-I diagnostics.
        self.ax_dop.clear(); self.ax_i.clear()
        self.ax_dop.imshow(result.dop_map(), cmap="viridis", vmin=0, vmax=1,
                           aspect="auto")
        self.ax_dop.set_title("DoP after reflection", fontsize=10)
        self.ax_dop.set_xticks([]); self.ax_dop.set_yticks([])
        im_i = self.ax_i.imshow(result.stokes_grid[..., 0], cmap="magma",
                                aspect="auto")
        self.ax_i.set_title("total I (no analyzer)", fontsize=10)
        self.ax_i.set_xticks([]); self.ax_i.set_yticks([])
        self.fig.colorbar(im_i, ax=self.ax_i, fraction=0.046, pad=0.04)

        # Analyzer sweep at the image center, driven by the same kernel.
        self.ax_sweep.clear()
        j, i = (s // 2 for s in result.stokes_grid.shape[:2])
        center = result.stokes_grid[j, i]
        angles = np.linspace(0.0, 180.0, 181)
        ext = float(self.sl_ext.val)
        sweep = [analyzer_intensity(center, a, ext) for a in angles]
        self.ax_sweep.plot(angles, sweep, lw=1.6)
        self.ax_sweep.axvline(float(self.sl_ana.val), color="r", ls="--",
                              lw=1.0, label="analyzer slider")
        theta_b = np.degrees(brewster_angle(1.0, float(n2.real)))
        self.ax_sweep.axvline(theta_b, color="g", ls=":", lw=1.0,
                              label=f"Brewster {theta_b:.1f} deg")
        self.ax_sweep.set_title("analyzer sweep, image center", fontsize=10)
        self.ax_sweep.set_xlabel("analyzer angle [deg]")
        self.ax_sweep.set_ylabel("intensity")
        self.ax_sweep.legend(fontsize=7, loc="upper right")
        self.ax_sweep.grid(alpha=0.3)

        # Center-pixel Stokes state text panel.
        self.ax_text.clear(); self.ax_text.axis("off")
        dop, dolp, docp = degrees_of_polarization(center)
        rep = check_feasible(center)
        psi = 0.5 * np.degrees(np.arctan2(center[2], center[1])) \
            if dolp > 1e-9 else float("nan")
        txt = (
            "center pixel Stokes (lens frame)\n"
            f"  I = {center[0]:.5f}\n"
            f"  Q = {center[1]:.5f}\n"
            f"  U = {center[2]:.5f}\n"
            f"  V = {center[3]:.5f}\n"
            f"  DoP  = {dop:.5f}\n"
            f"  DoLP = {dolp:.5f}\n"
            f"  DoCP = {docp:.5f}\n"
            f"  psi  = {psi:.2f} deg\n"
            f"  incidence = {result.incidence_deg[j, i]:.2f} deg\n"
            f"  feasible: {rep.ok}\n"
            f"  policy: {result.engine.policy}\n"
            f"  valid pixels: {int(result.valid.sum())}"
        )
        self.ax_text.text(0.0, 0.97, txt, va="top", family="monospace",
                          fontsize=9)
        self.fig.canvas.draw_idle()

    def _set_status(self, msg, ok):
        self.text_status.set_text(("" if ok else "!! ") + msg)
        self.text_status.set_color("#1d6b1d" if ok else "#7a1f1f")
        self.fig.canvas.draw_idle()


def main():
    # The widgets are matplotlib-native and work with any interactive
    # backend. If the environment only has a headless Agg build, report
    # clearly instead of crashing on an empty window.
    backend = matplotlib.get_backend()
    if backend.lower() == "agg":
        raise RuntimeError(
            "No interactive matplotlib backend is available (current: Agg). "
            "Install a GUI toolkit (python3-tk, PyQt5, ...) or run "
            "`python scripts/render_headless.py` to generate the same "
            "images without a window.")
    win = ExperimentWindow()
    plt.show()
    return win


if __name__ == "__main__":
    main()

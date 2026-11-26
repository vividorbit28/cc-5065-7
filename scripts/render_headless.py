"""Headless reproduction of the experiment window outputs.

Generates, under outputs/:
  * parallel/crossed/45/135 analyzer intensity images for a corrugated
    dielectric plane and for a double-bounce V-groove;
  * total-I and DoP diagnostic maps;
  * an analyzer-angle sweep showing the extinction contrast;
  * a text report with Stokes states, DoP ranges and feasibility verdicts;
  * an .npz archive with every raw array for re-plotting.

Run:  .venv/bin/python scripts/render_headless.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polrender import (
    CorrugatedPlane, VGroove, PolarizedRenderer, analyzer_intensity,
    brewster_angle, check_feasible, degrees_of_polarization, from_components,
    linear, zero,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")
os.makedirs(OUT, exist_ok=True)
ANALYZER_ANGLES = (0.0, 45.0, 90.0, 135.0)


def save_analyzer_stack(result, prefix, report):
    vmax = max(result.stokes_grid[..., 0].max(), 1e-9)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, ang, img in zip(axes.ravel(), result.analyzer_angles,
                            result.intensity):
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"{prefix}: analyzer {ang:g} deg")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{prefix} -- analyzer intensity (same grayscale scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{prefix}_analyzers.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].imshow(result.stokes_grid[..., 0], cmap="magma", aspect="auto")
    axes[0].set_title("total I (no analyzer)")
    im1 = axes[1].imshow(result.dop_map(), cmap="viridis", vmin=0, vmax=1,
                         aspect="auto")
    axes[1].set_title("DoP after reflection")
    # Stokes Q/U sign maps show the polarization axis rotating across view.
    qu = np.zeros((*result.stokes_grid.shape[:2], 3))
    q = result.stokes_grid[..., 1]
    u = result.stokes_grid[..., 2]
    scale = max(np.hypot(q, u).max(), 1e-12)
    qu[..., 0] = np.clip(q / scale, 0, 1)
    qu[..., 1] = np.clip(u / scale, 0, 1)
    qu[..., 2] = np.clip(-q / scale, 0, 1)
    axes[2].imshow(qu, aspect="auto")
    axes[2].set_title("Q (red) / U (green) sign map")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{prefix}_diagnostics.png"), dpi=130)
    plt.close(fig)

    j, i = (s // 2 for s in result.stokes_grid.shape[:2])
    center = result.stokes_grid[j, i]
    dop, dolp, docp = degrees_of_polarization(center)
    report.append(f"[{prefix}]")
    report.append(f"  valid pixels            : {int(result.valid.sum())}")
    report.append(f"  intensity range         : "
                  f"{result.intensity.min():.6f} .. {result.intensity.max():.6f}")
    report.append(f"  reflected I range       : "
                  f"{result.stokes_grid[..., 0].min():.6f} .. "
                  f"{result.stokes_grid[..., 0].max():.6f}")
    report.append(f"  per-pixel DoP range     : "
                  f"{result.dop_map()[result.valid].min():.6f} .. "
                  f"{result.dop_map()[result.valid].max():.6f}")
    report.append(f"  incidence range [deg]   : "
                  f"{result.incidence_deg[result.valid].min():.4f} .. "
                  f"{result.incidence_deg[result.valid].max():.4f}")
    report.append(f"  center Stokes (lens)    : "
                  f"I={center[0]:.6f} Q={center[1]:.6f} "
                  f"U={center[2]:.6f} V={center[3]:.6f}")
    report.append(f"  center DoP/DoLP/DoCP    : {dop:.6f} / {dolp:.6f} / {docp:.6f}")
    report.append(f"  center feasible         : {check_feasible(center).ok}")
    report.append(f"  kernel warnings         : "
                  f"{result.diagnostics if result.diagnostics else 'none'}")
    np.savez(os.path.join(OUT, f"{prefix}_arrays.npz"),
             stokes=result.stokes_grid, intensity=result.intensity,
             incidence=result.incidence_deg, valid=result.valid,
             analyzer_angles=result.analyzer_angles)
    return center, result


def main():
    report = []
    renderer_t = PolarizedRenderer("transport")
    renderer_r = PolarizedRenderer("reset")

    # 1. Fully linearly polarized source on the corrugated plane.
    plane = CorrugatedPlane(amplitude=0.35, period=2.0, n2=1.5)
    res_plane = renderer_t.render_corrugated_plane(
        plane, linear(1.0, 0.0), width=320, height=180, extent=1.0,
        analyzer_angles=ANALYZER_ANGLES)
    center_plane, _ = save_analyzer_stack(res_plane, "plane_linear", report)

    # 2. Partially polarized source (DoP=0.5) on the same plane.
    res_part = renderer_t.render_corrugated_plane(
        plane, from_components(1.0, dop=0.5, psi_deg=30.0),
        width=320, height=180, extent=1.0, analyzer_angles=ANALYZER_ANGLES)
    save_analyzer_stack(res_part, "plane_partial", report)

    # 3. Double-bounce V-groove.
    groove = VGroove(half_angle_deg=45.0, depth=1.0, n2=1.5)
    res_groove = renderer_t.render_v_groove(
        groove, from_components(1.0, dop=0.3, psi_deg=15.0),
        width=320, height=180, analyzer_angles=ANALYZER_ANGLES)
    save_analyzer_stack(res_groove, "vgroove", report)

    # 4. Exact no-light input: all outputs must be identically zero.
    res_dark = renderer_t.render_corrugated_plane(
        plane, zero(), width=80, height=40, extent=1.0,
        analyzer_angles=ANALYZER_ANGLES)
    assert np.all(res_dark.intensity == 0.0)
    assert np.all(res_dark.stokes_grid == 0.0)
    report.append("[dark-state control] all image and Stokes values exactly 0: OK")

    # 5. Degeneracy policy comparison on the near-normal column.
    res_reset = renderer_r.render_corrugated_plane(
        plane, linear(1.0, 0.0), width=320, height=180, extent=1.0,
        analyzer_angles=ANALYZER_ANGLES)
    delta = np.abs(res_reset.intensity - res_plane.intensity).max()
    report.append(f"[degeneracy policy] max |I_transport - I_reset| = {delta:.3e}")

    # 6. Analyzer sweep at the center pixel: extinction contrast.
    angles = np.linspace(0.0, 180.0, 361)
    sweep = np.array([analyzer_intensity(center_plane, a) for a in angles])
    i_max, i_min = sweep.max(), sweep.min()
    contrast = (i_max - i_min) / (i_max + i_min) if (i_max + i_min) else 0.0
    theta_b = np.degrees(brewster_angle(1.0, 1.5))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(angles, sweep, lw=1.8, label="center-pixel analyzer sweep")
    ax.axvline(angles[np.argmax(sweep)], color="g", ls="--", lw=1,
               label=f"max at {angles[np.argmax(sweep)]:.1f} deg")
    ax.axvline(angles[np.argmin(sweep)], color="r", ls="--", lw=1,
               label=f"min at {angles[np.argmin(sweep)]:.1f} deg")
    ax.axvline(theta_b, color="k", ls=":", lw=1,
               label=f"Brewster {theta_b:.2f} deg")
    ax.set_xlabel("analyzer angle [deg]")
    ax.set_ylabel("sensor intensity")
    ax.set_title(f"extinction contrast (Imax-Imin)/(Imax+Imin) = {contrast:.4f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "analyzer_sweep.png"), dpi=130)
    plt.close(fig)
    report.append(f"[analyzer sweep] Imax={i_max:.6f} Imin={i_min:.6f} "
                  f"contrast={contrast:.6f}")

    # 7. Reflected DoP versus incidence angle: use the partially polarized
    #    source so the reflection-induced rise of DoP is visible.
    row = res_part.stokes_grid.shape[0] // 2
    inc = res_part.incidence_deg[row]
    dop_row = res_part.dop_map()[row]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(inc, dop_row, lw=1.8, label="reflected DoP (source DoP=0.5)")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="incident DoP=0.5")
    ax.axvline(theta_b, color="r", ls="--", lw=1,
               label=f"Brewster {theta_b:.2f} deg")
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("reflected DoP")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Dielectric reflection raises the degree of polarization")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dop_vs_incidence.png"), dpi=130)
    plt.close(fig)

    # 8. Polarization path tracing and reference-frame diagnostics, rerun on
    #    the produced geometry to verify the localisation conclusion. The
    #    centre ray of each scene is traced stage by stage and checked
    #    against the independent world-coherency oracle.
    from polrender.diagnostics import (
        diagnose, format_report, compare_degeneracy_policies,
    )
    from polrender.frames import initial_frame

    diag_lines = ["", "=== polarization trace / reference-frame diagnosis ==="]
    # Corrugated plane centre ray.
    ops_plane = [("reflect", plane.normal_at(0.0), plane.n1, plane.n2),
                 ("lens",), ("analyze", 0.0, 0.0)]
    d_plane = diagnose(from_components(1.0, dop=1.0, psi_deg=30.0),
                       initial_frame(np.array([0.0, 0.0, 1.0])),
                       ops_plane, analyzer_angle=0.0)
    diag_lines.append("[corrugated plane, centre ray]")
    diag_lines.extend("  " + ln for ln in format_report(d_plane).splitlines())

    # V-groove two-bounce ray.
    ops_groove = [("reflect", groove.n_left, groove.n1, groove.n2),
                  ("reflect", groove.n_right, groove.n1, groove.n2),
                  ("lens",), ("analyze", 0.0, 0.0)]
    d_groove = diagnose(from_components(1.0, dop=0.3, psi_deg=15.0),
                        initial_frame(np.array([0.0, 0.0, -1.0])),
                        ops_groove, analyzer_angle=0.0)
    diag_lines.append("[V-groove, two-bounce ray]")
    diag_lines.extend("  " + ln for ln in format_report(d_groove).splitlines())

    # Measured degeneracy-policy choice (basis transport vs local reset).
    pol = compare_degeneracy_policies(
        from_components(1.0, dop=0.5, psi_deg=25.0),
        initial_frame(np.array([1e-7, 0.0, -1.0]) / np.linalg.norm([1e-7, 0.0, -1.0])),
        [("reflect", np.array([0.0, 0.0, 1.0]), 1.0, 1.5), ("lens",),
         ("analyze", 0.0, 0.0)])
    diag_lines.append(
        f"[degeneracy policy measurement] delta={pol['delta']:.3e} "
        f"equivalent={pol['equivalent']} recommended={pol['recommended']}")
    report.extend(diag_lines)

    # Save a per-stage trace table for the V-groove ray.
    with open(os.path.join(OUT, "polarization_trace.txt"), "w") as fh:
        fh.write("\n".join(diag_lines) + "\n")
        fh.write("\nV-groove per-stage trace\n")
        for st in d_groove["trace"].stages:
            rot = ("n/a" if st.basis_rotation_deg is None
                   else f"{st.basis_rotation_deg:+.3f}")
            fh.write(f"  #{st.index} {st.stage:9s} basis_rot={rot:>9s} deg "
                     f"I={st.intensity:.6f} DoP={st.dop:.6f} "
                     f"feasible={st.feasible}\n")

    report_text = "\n".join(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report_text + "\n")
    print(report_text)
    print(f"\nWrote images and arrays to {OUT}/")


if __name__ == "__main__":
    main()

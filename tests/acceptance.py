"""Analytical and numerical-boundary acceptance tests for the polarization kernel.

Run:  .venv/bin/python tests/acceptance.py
Exit code 0 means every physical check passed.
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from polrender import (
    PathEngine, PolarizedPath, CorrugatedPlane, VGroove, PolarizedRenderer,
    UnphysicalStokesError, analyzer_intensity, brewster_angle, check_feasible,
    degrees_of_polarization, from_components, initial_frame, linear,
    linear_polarizer, no_light_path, reflection, reflectance_normal,
    rotation, sanitize, transmission, zero,
)

TOL = 1e-9
failures = []
passed = []


def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  PASS  {name} {detail}")
    else:
        failures.append((name, detail))
        print(f"  FAIL  {name} {detail}")


def approx(a, b, rel=1e-9, abs_tol=1e-12):
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def section(title):
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------
section("1. Stokes feasibility: partial polarization, zero light, bad input")

s_part = from_components(2.0, dop=0.4, psi_deg=30.0)
rep = check_feasible(s_part)
check("partial polarization DoP=0.4", approx(rep.dop, 0.4, rel=1e-12),
      f"DoP={rep.dop}")
check("partial state feasible", rep.ok)

dop, dolp, docp = degrees_of_polarization(s_part)
check("DoLP equals DoP for linear input", approx(dolp, 0.4, rel=1e-12))
check("DoCP zero for linear input", approx(docp, 0.0, abs_tol=1e-15))

dark = zero()
rep0 = check_feasible(dark)
check("zero state feasible", rep0.ok and rep0.dop == 0.0)
dop0, _, _ = degrees_of_polarization(dark)
check("zero state DoP defined as 0 (no division by I)", dop0 == 0.0)
check("analyzer on dark stays dark", analyzer_intensity(dark, 37.0) == 0.0)

bad = np.array([1.0, 2.0, 0.0, 0.0])  # DoP = 2
rep_bad = check_feasible(bad)
check("DoP>1 rejected", not rep_bad.ok and any("DoP>1" in m for m in rep_bad.messages))
try:
    check_feasible(bad, raise_on_fail=True)
    check("raise on unphysical", False)
except UnphysicalStokesError:
    check("raise on unphysical", True)

try:
    sanitize(np.array([0.0, 0.5, 0.0, 0.0]))
    check("zero-I polarized state cannot be silently clamped", False)
except UnphysicalStokesError:
    check("zero-I polarized state cannot be silently clamped", True)

s_proj = sanitize(np.array([1.0 + 1e-12, 1.0, 0.0, 0.0]))
check("tiny numeric overflow projected back onto cone",
      check_feasible(s_proj).ok and approx(s_proj[0], 1.0 + 1e-12, rel=1e-11))

try:
    from_components(1.0, dop=1.2)
    check("constructor rejects DoP>1", False)
except ValueError:
    check("constructor rejects DoP>1", True)

# --------------------------------------------------------------------------
section("2. Rotation matrices: R(a)R(b)=R(a+b), periodicity, inverse")

for a, b in [(10.0, 25.0), (-30.0, 70.0), (45.0, 45.0)]:
    lhs = rotation(a) @ rotation(b)
    rhs = rotation(a + b)
    check(f"R({a})R({b})=R({a + b})", np.allclose(lhs, rhs, atol=1e-12))

check("R(180) identity on Stokes", np.allclose(rotation(180.0), np.eye(4), atol=1e-12))
check("R(-theta) = R(theta)^T",
      np.allclose(rotation(-33.3), rotation(33.3).T, atol=1e-12))

# --------------------------------------------------------------------------
section("3. Ideal polarizer: Malus law, parallel/crossed analyzers")

s0 = linear(1.0, 0.0)
for theta in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
    measured = analyzer_intensity(s0, theta)
    malus = math.cos(math.radians(theta)) ** 2
    check(f"Malus I(theta={theta:g})=cos^2", approx(measured, malus, rel=1e-12),
          f"got {measured:.9f} want {malus:.9f}")

check("parallel analyzer transmits full intensity",
      approx(analyzer_intensity(s0, 0.0), 1.0))
check("crossed analyzer gives exact extinction",
      analyzer_intensity(s0, 90.0) == 0.0)

# Unpolarized light through an ideal polarizer: I/2, independent of angle.
sun = from_components(1.0, dop=0.0)
vals = [analyzer_intensity(sun, t) for t in (0.0, 30.0, 60.0, 120.0)]
check("unpolarized -> I/2 at every angle",
      all(approx(v, 0.5, rel=1e-12) for v in vals), f"{vals}")

# Two crossed polarizers around unpolarized light: zero.
m_cross = linear_polarizer(90.0) @ linear_polarizer(0.0)
check("crossed polarizer pair is a zero matrix (first row)",
      np.allclose(m_cross[0], 0.0, atol=1e-12))

# Rotating the *input basis* versus the analyzer angle: physical equivalence.
for frame_ang in (0.0, 23.5, -51.0):
    s_rot = rotation(frame_ang) @ s0
    direct = analyzer_intensity(s_rot, 40.0)
    equivalent = analyzer_intensity(s0, 40.0 - frame_ang)
    check(f"frame rotation {frame_ang:g} deg equivalent to analyzer shift",
          approx(direct, equivalent, rel=1e-12))

# --------------------------------------------------------------------------
section("4. Fresnel interface: energy range, R+T=1, Brewster, normal incidence")

n1, n2 = 1.0, 1.5
for deg in range(0, 90, 5):
    cos_i = math.cos(math.radians(deg))
    mr = reflection(n1, n2, cos_i)
    mt = transmission(n1, n2, cos_i)
    r_s, r_p = mr[0, 0] + mr[0, 1], mr[0, 0] - mr[0, 1]
    t_s, t_p = mt[0, 0] + mt[0, 1], mt[0, 0] - mt[0, 1]
    check(f"R in [0,1] at {deg:g} deg", 0.0 - 1e-12 <= r_s <= 1.0 + 1e-12
          and 0.0 - 1e-12 <= r_p <= 1.0 + 1e-12)
    check(f"T in [0,1] at {deg:g} deg", -1e-12 <= t_s <= 1.0 + 1e-12
          and -1e-12 <= t_p <= 1.0 + 1e-12)
    check(f"energy R+T=1 (s and p) at {deg:g} deg",
          approx(r_s + t_s, 1.0, rel=1e-10) and approx(r_p + t_p, 1.0, rel=1e-10),
          f"Rs+Ts={r_s + t_s:.9f} Rp+Tp={r_p + t_p:.9f}")
    # Reflected and transmitted states must stay physical for any
    # *physical* input (basis vectors like (0,1,0,0) are not Stokes
    # states and must not be used as probes).
    rng = np.random.default_rng(42)
    for _ in range(16):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        dop = rng.uniform(0.0, 1.0)
        physical = from_components(rng.uniform(0.1, 3.0), dop=dop,
                                   psi_deg=rng.uniform(0, 180),
                                   chi_deg=rng.uniform(-45, 45))
        check_feasible(mr @ physical, raise_on_fail=True)
        check_feasible(mt @ physical, raise_on_fail=True)

r0 = reflectance_normal(n1, n2)
r0_expected = ((n1 - n2) / (n1 + n2)) ** 2
check("normal reflectance ((n1-n2)/(n1+n2))^2",
      approx(r0, r0_expected, rel=1e-12), f"{r0:.9f} vs {r0_expected:.9f}")
mr0 = reflection(n1, n2, 1.0)
check("normal incidence s/p degenerate (M diagonal blocks equal)",
      approx(mr0[0, 1], 0.0, abs_tol=1e-12) and approx(mr0[1, 0], 0.0, abs_tol=1e-12))

theta_b = math.degrees(brewster_angle(n1, n2))
mb = reflection(n1, n2, math.cos(math.radians(theta_b)))
r_p_b = mb[0, 0] - mb[0, 1]
check("Brewster: p reflectance vanishes", abs(r_p_b) < 1e-12,
      f"theta_B={theta_b:.4f} deg, Rp={r_p_b:.3e}")

# Unpolarized light reflected at Brewster becomes fully s-polarized.
refl_b = mb @ from_components(1.0, dop=0.0)
dop_b, dolp_b, _ = degrees_of_polarization(refl_b)
check("Brewster reflection is fully linearly polarized",
      approx(dop_b, 1.0, rel=1e-9) and approx(dolp_b, 1.0, rel=1e-9),
      f"DoP={dop_b:.9f}")

# Grazing incidence: reflectance tends to 1.
mr_graze = reflection(n1, n2, math.cos(math.radians(89.999)))
r_graze = mr_graze[0, 0] + mr_graze[0, 1]
check("grazing R_s -> 1", r_graze > 0.999, f"Rs={r_graze:.6f}")

# --------------------------------------------------------------------------
section("5. Observable invariance under arbitrary coordinate-axis rotation")

# Take a reflected path, then re-express the SAME physical state in a frame
# rotated by beta. The analyzer measured at lab angle alpha must be unchanged.
engine = PathEngine("transport")
normal = np.array([0.0, 0.0, 1.0])
k_in = np.array([0.3, 0.1, -np.sqrt(1 - 0.1)])
frame = initial_frame(k_in)
src = from_components(1.0, dop=0.55, psi_deg=20.0)
path = PolarizedPath(np.array(src), frame)
engine.reflect(path, normal, 1.0, 1.5)
s_ref = np.array(path.stokes)

for beta in (0.0, 17.0, 45.0, -83.0, 123.0):
    # R(beta) is the passive frame transform: rotating the reference
    # frame by +beta makes the same lab-fixed analyzer read at
    # alpha+beta in the new components.
    s_beta = rotation(beta) @ s_ref
    for alpha in (0.0, 30.0, 60.0, 90.0, 140.0):
        i_a = analyzer_intensity(s_ref, alpha)
        i_b = analyzer_intensity(s_beta, alpha + beta)
        if not approx(i_a, i_b, rel=1e-10):
            check(f"invariance beta={beta} alpha={alpha}", False,
                  f"{i_a} vs {i_b}")
            break
    else:
        check(f"observable invariant under axis rotation beta={beta:g}", True)

# Operator identity for the passive rotation: the analyzer matrix in the
# rotated frame, transformed back, equals the lab-frame analyzer.
for beta in (12.5, -40.0, 90.0):
    for alpha in (0.0, 22.5, 67.5, 110.0):
        lhs = rotation(-beta) @ linear_polarizer(alpha + beta) @ rotation(beta)
        rhs = linear_polarizer(alpha)
        if not np.allclose(lhs, rhs, atol=1e-12):
            check(f"polarizer/rotation operator identity b={beta} a={alpha}",
                  False)
            break
    else:
        check(f"operator identity for beta={beta:g}", True)

# --------------------------------------------------------------------------
section("6. Rotate path basis vs rotate analyzer (the two experiments)")

# (a) Rotating only the local path basis is a coordinate change. The same
#     physical analyzer, re-expressed in the rotated frame (alpha-beta),
#     must give identical brightness at every setting.
base = from_components(1.0, dop=0.7, psi_deg=10.0)
lab_angles = [0.0, 45.0, 90.0, 135.0]
for beta in (-60.0, -30.0, 30.0, 60.0, 90.0):
    moved = rotation(beta) @ base
    ok = all(
        approx(analyzer_intensity(base, a),
               analyzer_intensity(moved, a + beta), rel=1e-12)
        for a in lab_angles
    )
    check(f"basis-only rotation beta={beta:g} leaves brightness unchanged", ok)
# And the total intensity I is exactly invariant under any frame rotation.
for beta in (-60.0, 30.0, 90.0):
    moved = rotation(beta) @ base
    check(f"total I invariant under basis rotation beta={beta:g}",
          approx(moved[0], base[0], abs_tol=1e-15))

# (b) Rotating only the analyzer changes measured intensity per Malus-like
#     law; crossed position reproduces extinction deterministically.
s_pol = linear(1.0, 0.0)
i_par = analyzer_intensity(s_pol, 0.0)
i_cross = analyzer_intensity(s_pol, 90.0)
i_45 = analyzer_intensity(s_pol, 45.0)
check("analyzer rotation: parallel max", approx(i_par, 1.0))
check("analyzer rotation: crossed extinction", i_cross == 0.0)
check("analyzer rotation: 45 deg gives 0.5", approx(i_45, 0.5))
# Repeatability: two evaluations are bit-identical.
check("extinction difference reproducible",
      analyzer_intensity(s_pol, 90.0) == i_cross
      and analyzer_intensity(s_pol, 0.0) == i_par)

# Partially polarized light: crossed does not go fully dark, and the
# residual equals (1-DoP)/2 * I as required for a linear state.
sp = from_components(1.0, dop=0.6, psi_deg=0.0)
check("partial polarization crossed residual",
      approx(analyzer_intensity(sp, 90.0), 0.2, rel=1e-12))
check("partial polarization parallel transmission",
      approx(analyzer_intensity(sp, 0.0), 0.8, rel=1e-12))

# --------------------------------------------------------------------------
section("7. Near-normal degeneracy: transport vs reset policies, same observables")

normal = np.array([0.0, 0.0, 1.0])
k_grazing_axis = np.array([1e-7, 0.0, -1.0])
k_grazing_axis = k_grazing_axis / np.linalg.norm(k_grazing_axis)

for policy in ("transport", "reset"):
    eng = PathEngine(policy)
    frame = initial_frame(k_grazing_axis)
    p = PolarizedPath(linear(1.0, 25.0), frame)
    eng.reflect(p, normal, 1.0, 1.5)
    rep = check_feasible(p.stokes)
    intensities = [analyzer_intensity(p.stokes, a) for a in (0.0, 45.0, 90.0)]
    check(f"policy={policy}: reflected state feasible at degeneracy", rep.ok)
    check(f"policy={policy}: intensities within energy range",
          all(0.0 - 1e-12 <= x <= 1.0 + 1e-12 for x in intensities),
          f"{intensities}")

# The two policies differ only by a frame rotation at the degenerate bounce.
# Analyzer sweeps in each frame must give the same set of observables.
eng_t = PathEngine("transport")
eng_r = PathEngine("reset")
ft = initial_frame(k_grazing_axis)
fr = initial_frame(k_grazing_axis)
pt = PolarizedPath(from_components(1.0, dop=0.5, psi_deg=25.0), ft)
pr = PolarizedPath(from_components(1.0, dop=0.5, psi_deg=25.0), fr)
eng_t.reflect(pt, normal, 1.0, 1.5)
eng_r.reflect(pr, normal, 1.0, 1.5)
sweep_t = np.array([analyzer_intensity(pt.stokes, a) for a in range(0, 181, 5)])
sweep_r = np.array([analyzer_intensity(pr.stokes, a) for a in range(0, 181, 5)])
check("transport/reset policy: same max, min and sweep shape",
      approx(sweep_t.max(), sweep_r.max(), rel=1e-9)
      and approx(sweep_t.min(), sweep_r.min(), rel=1e-9)
      and approx(np.sum(sweep_t), np.sum(sweep_r), rel=1e-9),
      f"max {sweep_t.max():.9f}/{sweep_r.max():.9f} "
      f"min {sweep_t.min():.9f}/{sweep_r.min():.9f}")

# Exact normal incidence: no unique s axis; reflection matrix must be
# proportional to identity in the linear block.
m_norm = reflection(1.0, 1.5, 1.0)
check("exact normal reflection isotropic",
      np.allclose(m_norm, m_norm[0, 0] * np.eye(4), atol=1e-12))

# --------------------------------------------------------------------------
section("8. Multi-bounce V-groove: double reflection, dark handling, feasibility")

groove = VGroove(half_angle_deg=45.0, depth=1.0, n2=1.5)
renderer = PolarizedRenderer("transport")
src = from_components(1.0, dop=0.3, psi_deg=15.0)
res = renderer.render_v_groove(groove, src, width=48, height=12,
                               analyzer_angles=(0.0, 90.0))
check("V-groove produced valid pixels", res.valid.sum() > 0,
      f"{res.valid.sum()} valid")
check("V-groove all grid states feasible",
      all(check_feasible(res.stokes_grid[j, i]).ok
          for j in range(res.stokes_grid.shape[0])
          for i in range(res.stokes_grid.shape[1]) if res.valid[j, i]))
check("V-groove intensities in [0, incident I]",
      np.nanmax(res.intensity) <= 1.0 + 1e-12 and np.nanmin(res.intensity) >= -1e-12)
check("V-groove no diagnostics", not res.diagnostics, str(res.diagnostics[:3]))

dark_res = renderer.render_v_groove(groove, zero(), width=24, height=8,
                                    analyzer_angles=(0.0, 45.0, 90.0))
check("dark input yields exactly zero images",
      np.all(dark_res.intensity == 0.0) and np.all(dark_res.stokes_grid == 0.0))

# --------------------------------------------------------------------------
section("9. Corrugated plane renderer: parallel/crossed maps and angle sweep")

plane = CorrugatedPlane(amplitude=0.35, period=2.0, n2=1.5)
r = renderer.render_corrugated_plane(
    plane, linear(1.0, 0.0), width=64, height=16, extent=1.0,
    analyzer_angles=(0.0, 45.0, 90.0, 135.0),
)
check("plane render all states feasible",
      all(check_feasible(r.stokes_grid[j, i]).ok
          for j in range(r.stokes_grid.shape[0])
          for i in range(r.stokes_grid.shape[1]) if r.valid[j, i]))
check("plane incidence span includes near-normal", r.incidence_deg.min() < 2.0,
      f"min={r.incidence_deg.min():.3f}")
check("plane incidence span reaches oblique angles", r.incidence_deg.max() > 25.0,
      f"max={r.incidence_deg.max():.3f}")
check("plane analyzer maps in energy range",
      r.intensity.max() <= 1.0 + 1e-12 and r.intensity.min() >= -1e-12)
# Parallel + crossed analyzer intensities sum back to the reflected I
# (ideal polarizer: I_out(0)+I_out(90) = I for any linear-block state).
i0 = r.intensity[0][r.valid]
i90 = r.intensity[2][r.valid]
i_total = r.stokes_grid[r.valid][:, 0]
check("I_parallel + I_crossed = reflected I",
      np.allclose(i0 + i90, i_total, atol=1e-12))
# 45 and 135 likewise.
i45 = r.intensity[1][r.valid]
i135 = r.intensity[3][r.valid]
check("I_45 + I_135 = reflected I",
      np.allclose(i45 + i135, i_total, atol=1e-12))
# DoP of reflected light must stay inside [0,1] at every pixel.
dop_map = r.dop_map()[r.valid]
check("per-pixel DoP within [0,1]",
      float(dop_map.min()) >= -1e-12 and float(dop_map.max()) <= 1.0 + 1e-12,
      f"range {dop_map.min():.6f}..{dop_map.max():.6f}")

# Renderer must refuse an unphysical source before any pixel is computed.
try:
    renderer.render_corrugated_plane(plane, np.array([1.0, 2.0, 0.0, 0.0]),
                                     width=8, height=8)
    check("renderer rejects unphysical source", False)
except UnphysicalStokesError:
    check("renderer rejects unphysical source", True)

# --------------------------------------------------------------------------
section("10. Matrix multiplication order and chain audit")

# Physical order: polarizer(0) then polarizer(45) on unpolarized light.
# S_out = M45 M0 S; result I = 1/4 (half at first, half again at 45).
chain = linear_polarizer(45.0) @ linear_polarizer(0.0)
out = chain @ from_components(1.0, dop=0.0)
check("chain order M45*M0 on unpolarized gives 1/4",
      approx(out[0], 0.25, rel=1e-12), f"I={out[0]:.9f}")
wrong_order = linear_polarizer(0.0) @ linear_polarizer(45.0)
out_w = wrong_order @ from_components(1.0, dop=0.0)
check("reversed order gives same scalar here but events stay ordered",
      approx(out_w[0], 0.25, rel=1e-12))

eng = PathEngine()
p = PolarizedPath(linear(1.0, 0.0), initial_frame(np.array([0.0, 0.0, -1.0])))
eng.reflect(p, np.array([0.0, 0.0, 1.0]), 1.0, 1.5)
eng.enter_lens(p)
eng.analyze(p, 30.0)
kinds = [e.kind for e in p.events]
check("event chain recorded in physical order",
      kinds == ["reflect", "lens", "analyzer"], str(kinds))

# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"PASSED: {len(passed)}   FAILED: {len(failures)}")
if failures:
    print("\nFailures:")
    for name, detail in failures:
        print(f"  - {name} {detail}")
    sys.exit(1)
print("All acceptance checks passed.")

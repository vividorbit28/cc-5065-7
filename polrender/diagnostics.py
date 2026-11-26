"""Polarization path tracing and reference-frame diagnostics.

This module is the diagnostic counterpart to :mod:`polrender.path`. It

* traces a propagation recipe stage by stage, recording the local
  transverse basis ``(e1, e2, k)``, the basis rotation at each step, the
  Mueller matrix that was applied, and the Stokes state entering/leaving
  every stage;
* propagates the *same* physical process through an independent
  world-space coherency (Jones) oracle that never touches the ray-local
  frame-rotation bookkeeping -- so a discrepancy can be attributed to
  either the *transport* end (reflection / frame transport) or the
  *lens-readout* end (the coordinate change feeding the analyzer);
* classifies analyzer readout errors into sign (mirror), angular offset
  and scale (normalisation) using a small set of *known* linear
  polarizer probes, without trusting the readout under test;
* reverses an exactly-invertible readout segment and recomputes analyzer
  readings, which separates a sign error from a reference-frame error;
* emits an execution evidence chain (check -> expected -> observed) and
  bounds the amount of per-stage logging.

Design rules (kept deliberately strict so the diagnostics never
false-report):

* The oracle shares the *Fresnel amplitudes* with the kernel (those are
  validated independently in the acceptance tests) but computes state
  propagation through a world coherency matrix, i.e. with no local-frame
  rotations at all. A reference-frame error in the kernel therefore shows
  up as a divergence; a numerically correct kernel agrees to ~1e-16.
* States are never clamped to hide an invalid computation: every recorded
  state is checked against the physical Stokes cone and any gross
  violation is reported as an anomaly at that stage.
* Logging is bounded by ``max_stages``; beyond it only a summary count is
  kept, so reference-frame localisation stays cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .frames import frame_rotation_angle, lens_frame, normalize
from .mueller import linear_polarizer, rotation
from .path import PathEngine, PolarizedPath
from .stokes import check_feasible, degrees_of_polarization

TOL = 1.0e-9
# Tolerance for "kernel and oracle agree". Far below any physical effect
# (intensities are O(1)) yet above float64 round-off seen on good paths.
AGREE_TOL = 1.0e-9


# ---------------------------------------------------------------------------
# Independent world-space coherency oracle
# ---------------------------------------------------------------------------

def _fresnel_amplitudes(n1, n2, cos_i):
    """rs, rp amplitudes matching :func:`polrender.mueller.reflection`.

    Shared physics (validated in the acceptance tests), *not* shared frame
    bookkeeping -- that is the part the oracle must be independent of.
    """
    n1 = complex(n1)
    n2 = complex(n2)
    sin_i = math.sqrt(max(0.0, 1.0 - cos_i * cos_i))
    sin_t = (n1 / n2) * sin_i
    sin2_t = sin_t * sin_t
    cos_t = complex(math.sqrt(max(0.0, 1.0 - sin2_t.real)), 0.0)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    return complex(rs), complex(rp)


def reflect_jones(k, normal, n1, n2):
    """World-space 3x3 Jones operator of a specular reflection.

    Returns ``(M, k_out)`` with ``E_out = M @ E_in``. Built from the
    interface s/p vectors directly; no ray-local frame is consulted.
    """
    k = normalize(k)
    n = normalize(normal)
    cos_i = abs(float(np.dot(-k, n)))
    cos_i = min(1.0, max(0.0, cos_i))
    rs, rp = _fresnel_amplitudes(n1, n2, cos_i)
    s = np.cross(n, k)
    if np.linalg.norm(s) < 1.0e-12:
        # Exact normal incidence: any consistent transverse pair works.
        s = np.cross(np.array([1.0, 0.0, 0.0]), k)
        if np.linalg.norm(s) < 1.0e-12:
            s = np.cross(np.array([0.0, 1.0, 0.0]), k)
    s = s / np.linalg.norm(s)
    p = np.cross(k, s)
    k_out = normalize(k - 2.0 * np.dot(k, n) * n)
    s2 = np.cross(n, k_out)
    s2 = s2 / np.linalg.norm(s2)
    p2 = np.cross(k_out, s2)
    M = rs * np.outer(s2, s) + rp * np.outer(p2, p)
    return M, k_out


def coherency_from_stokes(s, e1, e2):
    i, q, u, v = np.asarray(s, dtype=float)
    j = np.array([[(i + q) / 2.0, (u - 1j * v) / 2.0],
                  [(u + 1j * v) / 2.0, (i - q) / 2.0]], dtype=complex)
    return (j[0, 0] * np.outer(e1, e1) + j[0, 1] * np.outer(e1, e2)
            + j[1, 0] * np.outer(e2, e1) + j[1, 1] * np.outer(e2, e2))


def stokes_from_coherency(j, e1, e2):
    """Stokes vector in the (e1, e2, k) basis from a world coherency matrix.

    With J_ab = <E_a E_b*> in that basis:
        I = J11+J22, Q = J11-J22, U = 2 Re J12, V = -2 Im J12,
    matching :mod:`polrender.stokes` (V>0 right-handed about e1 x e2 = k).
    """
    j11 = np.vdot(e1, j @ e1)
    j12 = np.vdot(e1, j @ e2)
    j21 = np.vdot(e2, j @ e1)
    j22 = np.vdot(e2, j @ e2)
    i = float((j11 + j22).real)
    q = float((j11 - j22).real)
    u = float((j12 + j21).real)
    v = float(-2.0 * j12.imag)
    return np.array([i, q, u, v])


def _oracle_after_reflects(source, frame, reflect_ops):
    """World-coherency states after each reflect op.

    ``reflect_ops`` is a list of ``(normal, n1, n2)`` in physical order.
    Returns a list of (world_jones_prefix, k_out) so the caller can
    express the oracle state in whatever frame it wants.
    """
    e1, e2, k = (normalize(x) for x in frame)
    j = coherency_from_stokes(source, e1, e2)
    out = []
    for normal, n1, n2 in reflect_ops:
        m, k = reflect_jones(k, normal, n1, n2)
        j = m @ j @ m.conj().T
        out.append((j, k))
    return out


# ---------------------------------------------------------------------------
# Stage-by-stage trace
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    index: int
    stage: str
    frame: tuple
    basis_rotation_deg: float | None
    matrix: np.ndarray | None
    state_in: np.ndarray
    state_out: np.ndarray
    intensity: float
    dop: float
    feasible: bool
    message: str = ""


@dataclass
class TraceRecord:
    stages: list = field(default_factory=list)
    omitted: int = 0
    max_stages: int = 64

    @property
    def truncated(self) -> bool:
        return self.omitted > 0

    def final(self):
        return self.stages[-1].state_out if self.stages else np.zeros(4)

    def log_cost(self) -> str:
        kept = len(self.stages)
        return f"{kept} stages logged, {self.omitted} summarised away"


def _basis_rotation_deg(frame_a, frame_b):
    ka, kb = normalize(frame_a[2]), normalize(frame_b[2])
    if abs(abs(float(np.dot(ka, kb))) - 1.0) > 1.0e-9:
        return None  # direction changed between frames; no common-pane angle
    return frame_rotation_angle(frame_a, frame_b)


def run_recipe(engine: PathEngine, source, frame, ops, max_stages: int = 64):
    """Execute ``ops`` on a fresh path, recording a bounded stage trace.

    Supported ops (tuples):
        ("reflect", normal, n1, n2)
        ("transmit", normal, n1, n2, refracted_direction)
        ("propagate", k_new)
        ("lens",)                       # enter camera/lens coordinates
        ("analyze", axis_deg, extinction)
    """
    path = PolarizedPath(np.array(source, dtype=float), frame)
    trace = TraceRecord(max_stages=max_stages)
    for kind, *rest in ops:
        before = np.array(path.stokes, dtype=float)
        frame_before = path.frame
        n_events = len(path.events)
        if kind == "reflect":
            engine.reflect(path, rest[0], rest[1], rest[2])
        elif kind == "transmit":
            engine.transmit(path, rest[0], rest[1], rest[2], rest[3])
        elif kind == "propagate":
            engine.propagate_to(path, rest[0])
        elif kind == "lens":
            engine.enter_lens(path, rest[0] if rest else None)
        elif kind == "analyze":
            engine.analyze(path, rest[0], rest[1] if len(rest) > 1 else 0.0)
        else:
            raise ValueError(f"unknown op {kind!r}")
        matrix = path.events[-1].matrix if len(path.events) > n_events else None
        report = check_feasible(path.stokes)
        stage = Stage(
            index=len(trace.stages),
            stage=kind,
            frame=path.frame,
            basis_rotation_deg=_basis_rotation_deg(frame_before, path.frame),
            matrix=None if matrix is None else np.array(matrix),
            state_in=before,
            state_out=np.array(path.stokes, dtype=float),
            intensity=float(path.stokes[0]),
            dop=report.dop,
            feasible=report.ok,
            message=path.events[-1].note if len(path.events) > n_events else "",
        )
        if len(trace.stages) < trace.max_stages:
            trace.stages.append(stage)
        else:
            trace.omitted += 1
    return path, trace


def _reflect_ops_from(ops):
    return [(rest[0], rest[1], rest[2])
            for kind, *rest in ops if kind == "reflect"]


# ---------------------------------------------------------------------------
# Analyzer readout: closed-form map, extinction, sign/angle/scale classify
# ---------------------------------------------------------------------------

def analyzer_intensity_map(stokes_grid, angle_deg, extinction: float = 0.0):
    """Vectorised closed form of ``linear_polarizer(angle)^T`` first row.

    I(a) = a I + b (Q cos2a + U sin2a), a=(1+e)/2, b=(1-e)/2. Independent
    of the matrix kernel, so it is a trustworthy reference curve.
    """
    s = np.asarray(stokes_grid, dtype=float)
    a0 = 0.5 * (1.0 + extinction)
    b0 = 0.5 * (1.0 - extinction)
    t = math.radians(angle_deg)
    return a0 * s[..., 0] + b0 * (s[..., 1] * math.cos(2 * t)
                                 + s[..., 2] * math.sin(2 * t))


def measure_extinction(readout, psi_deg: float, step_deg: float = 0.05):
    """Angle in [0,180) of minimum transmitted intensity for a linear probe.

    ``readout(state, angle)`` returns the measured intensity. Uses a coarse
    then local refinement scan so the cost is bounded and deterministic.
    """
    from .stokes import linear as _linear
    probe = _linear(1.0, psi_deg)
    coarse = np.arange(0.0, 180.0, 1.0)
    vals = np.array([readout(probe, a) for a in coarse])
    a0 = coarse[int(np.argmin(vals))]
    fine = np.arange(a0 - 1.0, a0 + 1.0 + 0.5 * step_deg, step_deg)
    vals = np.array([readout(probe, a) for a in fine])
    return float(fine[int(np.argmin(vals))]) % 180.0


def _ang_diff(a, b):
    """Signed difference a-b wrapped to (-90, 90]."""
    d = (a - b + 90.0) % 180.0 - 90.0
    return d


def classify_analyzer(readout, probes=(15.0, 40.0, 75.0), extinction: float = 0.0):
    """Classify a linear-analyzer readout as sign / angle / scale / ok.

    Two known linear polarizer probes pin the *mapping* from the physical
    extinction angle (psi+90) to the observed one: the mirror hypothesis
    predicts ``obs = -(psi+90)`` and the offset hypothesis predicts
    ``obs = (psi+90) + d``. A parallel-transmission ratio then flags a
    scale (normalisation) error. Returns a dict with ``verdict`` in
    {"ok", "sign", "angle", "scale", "convention"} plus the fitted gains.
    """
    ideal = []
    observed = []
    for psi in probes:
        ideal.append((psi + 90.0) % 180.0)
        observed.append(measure_extinction(readout, psi))
    ideal = np.array(ideal)
    observed = np.array(observed)

    # Fit observed = s * ideal + d (mod 180) for s = +1 (offset) and s = -1
    # (mirror). Residual is the max wrapped angular error.
    def residual(sign):
        d = np.array([_ang_diff(o, sign * idl) for o, idl in zip(observed, ideal)])
        d0 = float(np.median(d))
        res = max(abs(_ang_diff(o, (sign * idl + d0) % 180.0))
                  for o, idl in zip(observed, ideal))
        return d0, res

    d_pos, r_pos = residual(+1.0)
    d_neg, r_neg = residual(-1.0)

    # Parallel transmission ratio -> scale of the readout.
    from .stokes import linear as _linear
    ratios = []
    for psi in probes:
        obs_par = readout(_linear(1.0, psi), psi)  # parallel: full (ideal)
        if obs_par > TOL:
            ratios.append(obs_par)
    scale = float(np.median(ratios)) if ratios else float("nan")

    if r_pos < 0.2 and abs(d_pos) < 0.2:
        verdict = "ok"
        sign, offset = +1.0, d_pos
    elif r_neg < 0.2 and abs(d_neg) < 0.2:
        verdict = "sign"
        sign, offset = -1.0, d_neg
    elif r_pos < 0.2:
        verdict = "angle"
        sign, offset = +1.0, d_pos
    elif r_neg < 0.2:
        verdict = "angle"
        sign, offset = -1.0, d_neg
    else:
        verdict = "convention"
        sign, offset, r_pos = +1.0, 0.0, min(r_pos, r_neg)
    if verdict == "ok" and abs(scale - 1.0) > 1.0e-6:
        verdict = "scale"

    return {
        "verdict": verdict,
        "sign": sign,
        "offset_deg": float(offset),
        "scale": scale,
        "residual_deg": float(r_pos if sign > 0 else r_neg),
        "ideal_extinctions": ideal.tolist(),
        "observed_extinctions": observed.tolist(),
    }


# ---------------------------------------------------------------------------
# Reverse a readout segment: sign error vs reference-frame error
# ---------------------------------------------------------------------------

def reverse_readout_check(stokes_lens, applied_angle_deg, geometric_angle_deg,
                          readout, probe_angles=(0.0, 45.0, 90.0, 135.0)):
    """Reverse the lens rotation and separate a sign error from a frame error.

    Parameters
    ----------
    stokes_lens : ray state after the pipeline applied a lens rotation.
    applied_angle_deg : the rotation the pipeline actually used.
    geometric_angle_deg : the correct ray->lens rotation from frame
        transport. A frame error means ``applied != geometric``.
    readout : ``readout(state, angle) -> intensity`` (the analyzer under
        test, e.g. the matrix kernel).

    Physics
    -------
    The pipeline forms ``S_lens = R(applied) S_ray`` then reads
    ``readout(S_lens, a)``. Reversing with the *geometric* angle recovers
    ``S_ray = R(-geometric) S_lens``. With a correct analyzer and correct
    frame:

        readout(S_ray, a) == closed_form(S_ray, a)            (sign ok)
        readout(S_lens, a) == closed_form(S_ray, a + applied)  (frame ok)

    A *sign* (mirror) error lives in the analyzer operator itself, so it
    makes ``readout(S_ray, a) == closed_form(S_ray, -a)`` regardless of the
    frame. A *frame* error leaves the reversed reading correct but shifts
    the forward reading by ``applied - geometric``. The two signatures are
    therefore separable.
    """
    s_lens = np.asarray(stokes_lens, dtype=float)
    s_ray = rotation(-geometric_angle_deg) @ s_lens
    angles = np.asarray(probe_angles, dtype=float)
    ref = np.array([analyzer_intensity_map(s_ray, a) for a in angles])
    ref_mirror = np.array([analyzer_intensity_map(s_ray, -a) for a in angles])
    rev = np.array([float(readout(s_ray, a)) for a in angles])
    fwd = np.array([float(readout(s_lens, a)) for a in angles])
    expected_fwd = np.array([analyzer_intensity_map(s_ray, a + applied_angle_deg)
                             for a in angles])

    sign_ok = float(np.max(np.abs(rev - ref)))
    sign_mirror = float(np.max(np.abs(rev - ref_mirror)))
    frame_res = float(np.max(np.abs(fwd - expected_fwd)))

    # Best-fit extra rotation of the forward sweep vs the applied angle.
    best_shift, best_res = 0.0, float("inf")
    for d in np.arange(-90.0, 90.0, 0.1):
        pred = np.array([analyzer_intensity_map(s_ray, a + applied_angle_deg + d)
                         for a in angles])
        r = float(np.max(np.abs(fwd - pred)))
        if r < best_res:
            best_res, best_shift = r, float(d)

    if sign_ok < 1.0e-9 and frame_res < 1.0e-9:
        verdict = "ok"
    elif sign_mirror < 1.0e-9 and sign_ok > 1.0e-6:
        verdict = "sign"
    elif sign_ok < 1.0e-9 and best_res < 1.0e-9 and abs(best_shift) > 1.0e-6:
        verdict = "frame"
    else:
        verdict = "inconsistent"
    return {
        "verdict": verdict,
        "frame_shift_deg": float(best_shift),
        "sign_residual": sign_ok,
        "mirror_residual": sign_mirror,
        "frame_residual": frame_res,
        "forward": fwd.tolist(),
        "reversed": rev.tolist(),
        "reference": ref.tolist(),
    }


# ---------------------------------------------------------------------------
# Full diagnosis: stage-localised evidence chain
# ---------------------------------------------------------------------------

def diagnose(source, frame, ops, policy: str = "transport",
             analyzer_angle: float = 0.0, extinction: float = 0.0,
             max_stages: int = 64):
    """Trace ``ops`` and produce a stage-localised evidence chain.

    Returns a dict with the bounded trace, per-stage oracle comparison,
    lens-readout check, extinction classification and an overall cause in
    {"none", "transport", "lens-readout", "state-invalid"}.
    """
    engine = PathEngine(policy)
    path, trace = run_recipe(engine, source, frame, ops, max_stages=max_stages)

    evidence = []

    def add(check, expected, observed, ok):
        evidence.append({"check": check, "expected": expected,
                         "observed": observed, "ok": bool(ok)})

    add("all stages physical", "feasible at every stage",
        "infeasible: " + ",".join(
            f"#{s.index}:{s.stage}" for s in trace.stages if not s.feasible),
        all(s.feasible for s in trace.stages))
    state_valid = all(s.feasible for s in trace.stages)

    # Independent oracle at every reflection stage.
    reflect_ops = _reflect_ops_from(ops)
    oracle = _oracle_after_reflects(np.array(source, dtype=float), frame,
                                    reflect_ops)
    ref_stage_indices = [s.index for s in trace.stages if s.stage == "reflect"]
    worst = (0.0, None)
    first_div = None
    for (j, k_out), stage_idx in zip(oracle, ref_stage_indices):
        stage = next(s for s in trace.stages if s.index == stage_idx)
        e1, e2 = stage.frame[0], stage.frame[1]
        s_oracle = stokes_from_coherency(j, e1, e2)
        resid = float(np.abs(s_oracle - stage.state_out).max())
        if resid > worst[0]:
            worst = (resid, stage_idx)
        if resid > AGREE_TOL and first_div is None:
            first_div = stage_idx
    transport_ok = first_div is None
    add("transport matches world-coherency oracle",
        f"max residual < {AGREE_TOL:g}",
        f"max residual {worst[0]:.3e} at stage {worst[1]}", transport_ok)

    # Lens-readout check: the recorded 'lens' rotation must equal the true
    # ray->lens frame rotation of the *previous* frame.
    lens_ok = True
    lens_detail = "no lens stage"
    lens_shift = 0.0
    for idx, stage in enumerate(trace.stages):
        if stage.stage != "lens":
            continue
        prev = trace.stages[idx - 1] if idx > 0 else None
        if prev is None or stage.matrix is None:
            continue
        expected = rotation(_basis_rotation_deg(prev.frame, stage.frame) or 0.0)
        diff = np.abs(expected - stage.matrix).max()
        lens_ok = lens_ok and diff < 1.0e-9
        _, ang_expected = lens_frame(prev.frame)
        lens_shift = _ang_diff(stage.basis_rotation_deg or 0.0, ang_expected)
        lens_detail = (f"|applied - correct rotation|={diff:.3e}, "
                       f"shift={lens_shift:+.3f} deg")
    add("lens readout rotation matches frame transport", "difference ~ 0",
        lens_detail, lens_ok)

    # Extinction classification against the closed-form reference.
    def kernel_readout(state, ang):
        out = linear_polarizer(ang, extinction) @ np.asarray(state, float)
        return float(out[0])

    cls = classify_analyzer(kernel_readout, extinction=extinction)
    add("analyzer readout convention (sign/angle/scale)",
        "ok, sign=+1, offset=0, scale=1",
        f"{cls['verdict']}, sign={cls['sign']:+.0f}, "
        f"offset={cls['offset_deg']:+.3f} deg, scale={cls['scale']:.6f}",
        cls["verdict"] == "ok")

    # Reverse-readout: separates a sign error from a frame error.
    lens_stage = next((s for s in trace.stages if s.stage == "lens"), None)
    if lens_stage is not None:
        prev = trace.stages[trace.stages.index(lens_stage) - 1] \
            if trace.stages.index(lens_stage) > 0 else None
        geometric = _basis_rotation_deg(prev.frame, lens_stage.frame) \
            if prev is not None else 0.0
        rev = reverse_readout_check(
            lens_stage.state_out,
            lens_stage.basis_rotation_deg or 0.0,
            geometric or 0.0,
            kernel_readout,
        )
    else:
        rev = {"verdict": "ok", "frame_shift_deg": 0.0,
               "sign_residual": 0.0, "mirror_residual": 0.0,
               "frame_residual": 0.0}
    add("reverse readout separates sign from frame error",
        "ok (no frame shift, not mirrored)",
        f"{rev['verdict']}, frame_shift={rev['frame_shift_deg']:+.3f} deg",
        rev["verdict"] == "ok")

    if not state_valid:
        cause = "state-invalid"
    elif not transport_ok:
        cause = "transport"
    elif not lens_ok or cls["verdict"] != "ok" or rev["verdict"] != "ok":
        cause = "lens-readout"
    else:
        cause = "none"

    center = trace.final()
    dop, dolp, docp = degrees_of_polarization(center)
    return {
        "policy": policy,
        "cause": cause,
        "trace": trace,
        "path": path,
        "evidence": evidence,
        "oracle_worst_residual": worst[0],
        "oracle_worst_stage": worst[1],
        "first_diverging_stage": first_div,
        "lens_shift_deg": lens_shift,
        "classification": cls,
        "reverse": rev,
        "final_stokes": center,
        "final_dop": dop,
        "final_dolp": dolp,
        "final_docp": docp,
        "log_cost": trace.log_cost(),
    }


def format_report(diag: dict) -> str:
    """Human-readable evidence chain for a diagnosis dict."""
    lines = []
    lines.append(f"policy={diag['policy']}  cause={diag['cause']}  "
                 f"({diag['log_cost']})")
    for ev in diag["evidence"]:
        mark = "OK " if ev["ok"] else "BAD"
        lines.append(f"  [{mark}] {ev['check']}  ({ev['observed']})")
    c = diag["classification"]
    lines.append(f"  extinction: ideal={np.round(c['ideal_extinctions'], 3).tolist()} "
                 f"observed={np.round(c['observed_extinctions'], 3).tolist()}")
    r = diag["reverse"]
    lines.append(f"  reverse-readout: {r['verdict']} "
                 f"(frame shift {r['frame_shift_deg']:+.3f} deg)")
    i = diag["final_stokes"]
    lines.append(f"  final I={i[0]:.6f} Q={i[1]:.6f} U={i[2]:.6f} V={i[3]:.6f} "
                 f"DoP={diag['final_dop']:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Degeneracy policy measured choice
# ---------------------------------------------------------------------------

def compare_degeneracy_policies(source, frame, ops, probes=(0.0, 45.0, 90.0)):
    """Measure whether transport and reset frames give identical observables.

    The choice between carrying the previous basis and rebuilding a local
    orthonormal frame at each interaction is made from this measurement,
    not from convention: if both policies yield the same analyzer sweep
    within tolerance they are the same physics and the more stable
    (transport) is recommended through the degenerate bounce.
    """
    results = {}
    for policy in ("transport", "reset"):
        engine = PathEngine(policy)
        path, _ = run_recipe(engine, source, frame, ops)
        results[policy] = path.stokes
    sweep = {}
    for policy, s in results.items():
        sweep[policy] = np.array([analyzer_intensity_map(s, a) for a in probes])
    delta = float(np.abs(sweep["transport"] - sweep["reset"]).max()) \
        if set(sweep) == {"transport", "reset"} else float("inf")
    return {
        "delta": delta,
        "equivalent": delta < AGREE_TOL,
        "recommended": "transport" if delta < AGREE_TOL else "reset",
        "sweeps": {k: v.tolist() for k, v in sweep.items()},
    }

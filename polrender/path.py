"""End-to-end polarized path propagation.

A :class:`PolarizedPath` carries a Stokes state together with the frame it
is expressed in. Every optical event records the matrices applied so the
full chain  S_out = M_n ... M_1 S_in  is auditable, and every intermediate
state is re-checked against the physical Stokes cone before color/output
conversion may consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import (
    transport_propagate,
    transport_reflect,
    frame_rotation_angle,
    lens_frame,
    normalize,
)
from .mueller import reflection, transmission, linear_polarizer, rotation
from .stokes import sanitize, check_feasible, zero, FeasibilityReport


@dataclass
class Event:
    kind: str
    matrix: np.ndarray
    frame: tuple
    note: str = ""


@dataclass
class PolarizedPath:
    stokes: np.ndarray
    frame: tuple
    events: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def report(self) -> FeasibilityReport:
        return check_feasible(self.stokes)

    def apply(self, matrix: np.ndarray, new_frame: tuple, kind: str, note: str = ""):
        rotated = matrix @ self.stokes
        try:
            rotated = sanitize(rotated)
        except ValueError as exc:
            self.warnings.append(f"{kind}: {exc}")
            raise
        self.stokes = rotated
        self.frame = new_frame
        self.events.append(Event(kind, np.array(matrix), new_frame, note))
        report = check_feasible(self.stokes)
        if not report.ok:
            self.warnings.append(f"{kind}: {report.summary()}")
        return self


class PathEngine:
    """Builds reflected/transmitted paths with explicit frame bookkeeping."""

    def __init__(self, degeneracy_policy: str = "transport"):
        if degeneracy_policy not in ("transport", "reset"):
            raise ValueError("degeneracy_policy must be 'transport' or 'reset'")
        self.policy = degeneracy_policy

    def propagate_to(self, path: PolarizedPath, k_new) -> PolarizedPath:
        new_frame = transport_propagate(path.frame, k_new)
        angle = frame_rotation_angle(path.frame, new_frame)
        # Free flight changes nothing physically; the state is re-expressed
        # in the new frame via the passive Mueller rotation R(angle).
        path.apply(rotation(angle), new_frame, "propagate",
                   f"frame rotation {angle:.4f} deg")
        return path

    def reflect(self, path: PolarizedPath, normal, n1, n2) -> PolarizedPath:
        normal = normalize(normal)
        k = normalize(path.frame[2])
        cos_i = abs(float(np.dot(-k, normal)))
        cos_i = min(1.0, max(0.0, cos_i))
        tr = transport_reflect(path.frame, normal, policy=self.policy)
        # Interface matrix is diagonal in the (s,p) frame, which the new
        # frame coincides with; re-express the state in that frame first.
        r_frame = rotation(tr.angle_from_old_e1_deg)
        m_refl = reflection(n1, n2, cos_i)
        chain = m_refl @ r_frame
        path.apply(chain, tr.frame, "reflect",
                   f"theta_i={np.degrees(np.arccos(cos_i)):.3f} deg "
                   f"basis_rotation={tr.angle_from_old_e1_deg:.3f} deg"
                   f"{' [degenerate]' if tr.degenerate else ''}")
        return path

    def transmit(self, path: PolarizedPath, normal, n1, n2,
                 refracted_direction) -> PolarizedPath:
        normal = normalize(normal)
        k = normalize(path.frame[2])
        cos_i = abs(float(np.dot(-k, normal)))
        # Rebuild the transmitted frame from the same incidence plane.
        from .frames import interface_s_frame
        s = interface_s_frame(k, normal)
        k_t = normalize(refracted_direction)
        if s is None:
            if self.policy == "transport":
                e1 = path.frame[0] - k_t * np.dot(path.frame[0], k_t)
                e1 = normalize(e1)
            else:
                anchor = np.array([1.0, 0.0, 0.0])
                e1 = anchor - k_t * np.dot(anchor, k_t)
                e1 = normalize(e1)
            degenerate = True
        else:
            e1 = s
            degenerate = False
        e2 = normalize(np.cross(k_t, e1))
        new_frame = (e1, e2, k_t)
        angle = frame_rotation_angle(path.frame, new_frame)
        m_trans = transmission(n1, n2, min(1.0, max(0.0, cos_i)))
        chain = m_trans @ rotation(angle)
        path.apply(chain, new_frame, "transmit",
                   f"degenerate={degenerate}")
        return path

    def enter_lens(self, path: PolarizedPath, up_hint=None) -> tuple:
        """Transform the path state into camera/lens coordinates.

        Returns (path, lens_frame, angle_deg). No optics yet, only the
        coordinate change, so analyzer experiments can rotate either the
        local path basis (invariant) or the analyzer (physical effect).
        """
        lf, angle = lens_frame(path.frame, up_hint=up_hint)
        path.apply(rotation(angle), lf, "lens",
                   f"ray->lens frame rotation {angle:.4f} deg")
        return path, lf, angle

    def analyze(self, path: PolarizedPath, analyzer_angle_deg: float,
                extinction: float = 0.0) -> float:
        """Virtual rotatable analyzer in front of the sensor integration."""
        m = linear_polarizer(analyzer_angle_deg, extinction)
        before = np.array(path.stokes)
        out = m @ before
        out = sanitize(out)
        path.stokes = out
        path.events.append(Event("analyzer", m, path.frame,
                                 f"axis={analyzer_angle_deg} deg"))
        return float(out[0])


def no_light_path(frame) -> PolarizedPath:
    """Explicit fully-dark state; DoP defined as zero and analyzer gives 0."""
    return PolarizedPath(zero(), frame)

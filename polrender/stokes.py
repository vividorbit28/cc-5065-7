"""Stokes vectors and physical-feasibility enforcement.

Conventions
-----------
Stokes column vectors are 4-vectors  S = (I, Q, U, V)^T.

* I : total intensity (W/m^2-like irradiance units).
* Q, U : linear polarization in the local transverse frame (e1, e2).
  - Q > 0 means linear polarization along e1; Q < 0 along e2.
  - U describes the +/-45 degree components.
* V : circular component, V > 0 right-handed with respect to +k
  (e1 x e2 = k).

A physical Stokes vector obeys
    I >= 0  and  Q^2 + U^2 + V^2 <= I^2.
The degree of polarization is  DoP = sqrt(Q^2+U^2+V^2) / I.
The degree of linear polarization is DoLP = sqrt(Q^2+U^2) / I.
The all-zero state S = 0 ("no light at all") is a valid physical state
and is handled explicitly: DoP is defined as 0 for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class UnphysicalStokesError(ValueError):
    """Raised when a Stokes state violates the physical cone by too much."""


@dataclass
class FeasibilityReport:
    ok: bool
    intensity: float
    dop: float
    dolp: float
    docp: float
    residual: float          # sqrt(Q^2+U^2+V^2) - I, <= 0 when physical
    messages: list

    def summary(self) -> str:
        if not self.ok:
            return "UNPHYSICAL: " + "; ".join(self.messages)
        return (
            f"I={self.intensity:.6g} DoP={self.dop:.6f} "
            f"DoLP={self.dolp:.6f} DoCP={self.docp:.6f} (feasible)"
        )


def as_stokes(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"Stokes vector must have shape (4,), got {arr.shape}")
    return arr


def zero() -> np.ndarray:
    """The unique no-light state."""
    return np.zeros(4)


def from_components(intensity, dop=0.0, psi_deg=0.0, chi_deg=0.0) -> np.ndarray:
    """Build a Stokes vector from intensity, DoP and orientation/ellipticity.

    psi : orientation of the major axis of the polarization ellipse [deg]
    chi : ellipticity angle [deg], chi=0 linear, |chi|=45 deg circular.
    """
    if intensity < 0:
        raise ValueError("intensity must be non-negative")
    if not (0.0 <= dop <= 1.0 + 1e-12):
        raise ValueError(f"DoP out of [0,1]: {dop}")
    dop = min(dop, 1.0)
    psi = math.radians(psi_deg)
    chi = math.radians(chi_deg)
    pol = intensity * dop
    return np.array([
        intensity,
        pol * math.cos(2.0 * chi) * math.cos(2.0 * psi),
        pol * math.cos(2.0 * chi) * math.sin(2.0 * psi),
        pol * math.sin(2.0 * chi),
    ])


def linear(intensity, psi_deg) -> np.ndarray:
    """Fully linearly polarized light with axis angle psi [deg] from e1."""
    return from_components(intensity, dop=1.0, psi_deg=psi_deg, chi_deg=0.0)


def degrees_of_polarization(s: np.ndarray):
    """Return (DoP, DoLP, DoCP). Zero light -> all zeros, no division."""
    s = np.asarray(s, dtype=float)
    i = s[0]
    if i <= 0.0:
        if np.any(s[1:] != 0.0):
            raise UnphysicalStokesError(
                f"zero/negative intensity I={i} but non-zero (Q,U,V)={s[1:]}"
            )
        return 0.0, 0.0, 0.0
    lin = math.hypot(s[1], s[2])
    dop = math.hypot(lin, s[3]) / i
    return dop, lin / i, abs(s[3]) / i


def check_feasible(s: np.ndarray, tol: float = 1e-9,
                   raise_on_fail: bool = False) -> FeasibilityReport:
    """Check that a state lies inside the physical Stokes cone."""
    s = np.asarray(s, dtype=float)
    i, q, u, v = s
    messages = []
    if i < -tol:
        messages.append(f"negative intensity I={i:.6g}")
    pol = math.sqrt(q * q + u * u + v * v)
    residual = pol - max(i, 0.0)
    if residual > tol * max(1.0, abs(i)):
        messages.append(
            f"DoP>1: |(Q,U,V)|={pol:.6g} exceeds I={i:.6g} "
            f"(DoP={pol / i if i else float('inf'):.6f})"
        )
    report = FeasibilityReport(
        ok=not messages,
        intensity=i,
        dop=(pol / i if i > 0 else 0.0),
        dolp=(math.hypot(q, u) / i if i > 0 else 0.0),
        docp=(abs(v) / i if i > 0 else 0.0),
        residual=residual,
        messages=messages,
    )
    if raise_on_fail and not report.ok:
        raise UnphysicalStokesError(report.summary())
    return report


def sanitize(s: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Repair *numerical-noise* cone violations only; never mask real errors.

    The true zero state is preserved. A state whose polarized magnitude
    exceeds its intensity by more than ``tol`` is an error of the matrix
    chain (a genuine DoP>1), not a rounding artefact, and is raised rather
    than silently projected onto the cone. Likewise a non-positive
    intensity with non-zero polarized components is an error. This keeps
    invalid computations observable in the diagnostics instead of being
    clamped away.
    """
    s = np.array(s, dtype=float)
    i, q, u, v = s
    pol = math.hypot(math.hypot(q, u), v)
    if i <= tol:
        if pol > tol:
            raise UnphysicalStokesError(
                "cannot sanitize: non-positive intensity with non-zero "
                f"polarized components, I={i:.6g}, |QUV|={pol:.6g}"
            )
        return zero()
    excess = pol - i
    if excess > tol * max(1.0, abs(i)):
        raise UnphysicalStokesError(
            "cannot sanitize: DoP>1 beyond numerical noise, "
            f"I={i:.6g}, |QUV|={pol:.6g} (DoP={pol / i:.6f}); "
            "this indicates a broken matrix chain, not rounding"
        )
    if excess > 0.0:
        # Within numerical noise only: project onto the cone boundary.
        scale = i / pol
        return np.array([i, q * scale, u * scale, v * scale])
    return s

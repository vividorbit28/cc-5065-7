"""Mueller matrices: rotation, ideal polarizer, dielectric interface.

All matrices act on Stokes column vectors  S' = M @ S.
Chained optics applied in physical order M1 then M2 give  S_out = M2 @ M1 @ S_in.

Frame convention: every ray carries an orthonormal transverse basis
(e1, e2) with e1 x e2 = k (propagation direction). Angles are measured
from e1 toward e2, positive right-handed about k.
"""

from __future__ import annotations

import math

import numpy as np


def rotation(theta_deg: float) -> np.ndarray:
    """Standard Mueller coordinate-rotation matrix R(theta) (passive).

    When the reference frame itself is rotated by +theta (right-handed
    about k) from (e1,e2) to (e1',e2'), the same physical state's
    components in the new frame are  S' = R(theta) S. Consistency:
    a state polarized at +theta in the old frame reads Q'=1, U'=0 in
    the rotated frame -- applying R(theta) to (1, cos2theta, sin2theta)
    yields exactly (1,1,0,0). The inverse transform is R(-theta)=R^T.
    """
    t = math.radians(theta_deg)
    c = math.cos(2.0 * t)
    s = math.sin(2.0 * t)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,   c,   s, 0.0],
        [0.0,  -s,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def linear_polarizer(theta_deg: float, extinction: float = 0.0) -> np.ndarray:
    """Ideal linear polarizer with transmission axis at +theta from e1.

    extinction in [0, 1): transmitted intensity of the blocked eigenstate
    (0 = perfect polarizer). Rotating the *device* by theta means the
    diagonal principal-frame matrix m0 (axis along e1) is conjugated by
    the coordinate rotation:

        P(theta) = R(theta)^T @ m0 @ R(theta)

    i.e. P(theta) is the passive-frame version of "rotate the optic into
    place". Its first row is (a, b cos2theta, b sin2theta, 0), so a state
    linearly polarized at psi reads  I = a I + b |P| (cos2theta cos2psi
    + sin2theta sin2psi) = a I + b |P| cos2(psi-theta): the device passes
    light polarized parallel to +theta and extinguishes the orthogonal
    axis at theta+90, for every psi (not only the psi=0 axis).

    Using R(theta) m0 R(theta)^T instead would place the axis at -theta
    (a mirror about e1). That bug is invisible for psi in {0, 90} but
    swaps the 45/135 channels and mirrors every extinction position; it
    is exercised explicitly by the diagnostics and acceptance tests.
    """
    tmax = 1.0
    tmin = float(extinction)
    a = (tmax + tmin) / 2.0
    b = (tmax - tmin) / 2.0
    m0 = np.array([
        [a, b, 0.0, 0.0],
        [b, a, 0.0, 0.0],
        [0.0, 0.0, tmin, 0.0],
        [0.0, 0.0, 0.0, tmin],
    ])
    r = rotation(theta_deg)
    return r.T @ m0 @ r


def analyzer_intensity(s: np.ndarray, theta_deg: float,
                       extinction: float = 0.0) -> float:
    """Intensity measured behind an ideal linear analyzer at angle theta."""
    out = linear_polarizer(theta_deg, extinction) @ np.asarray(s, dtype=float)
    return float(out[0])


def _fresnel_amplitude(n1: complex, n2: complex, cos_i: float):
    """Fresnel amplitude coefficients.

    n1, n2 : (possibly complex) refractive indices.
    cos_i  : cosine of incidence angle inside medium 1, >= 0.
    Returns (rs, rp) as complex amplitudes with the standard sign
    convention; the Mueller construction below only uses |r|^2, Re/Im
    products, so global sign choices cancel in observables.
    """
    n1 = complex(n1)
    n2 = complex(n2)
    sin_i = math.sqrt(max(0.0, 1.0 - cos_i * cos_i))
    # Snell: sin_t = (n1/n2) sin_i (works with complex n2).
    sin_t = (n1 / n2) * sin_i
    sin2_t = sin_t * sin_t
    if abs(sin2_t.imag) < 1e-12 and sin2_t.real <= 1.0:
        # Lossless propagation: take the real positive root explicitly
        # instead of the complex principal sqrt (whose branch flips signs).
        cos_t = complex(math.sqrt(max(0.0, 1.0 - sin2_t.real)), 0.0)
    else:
        cos_t = np.sqrt(1.0 - sin2_t)  # absorbing media / evanescent wave
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    # Flipped-sign p coefficient for the reflected frame convention
    # (e2 = k x s after reflection): rs and rp share a sign at normal
    # incidence, making the normal-incidence reflection isotropic. The
    # flipped sign is a pure basis convention; |rp| is unchanged.
    rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    # Transmitted E amplitudes use the standard (unflipped) definitions,
    # never the 1+r identity, which only holds for the matching rp sign.
    ts = 2.0 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    tp = 2.0 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    return complex(rs), complex(rp), complex(cos_t), complex(ts), complex(tp)

def reflection(n1: complex, n2: complex, cos_i: float) -> np.ndarray:
    """Mueller matrix of specular reflection, in the local (s, p) frame.

    Row 0 = (s+s-, s-s-, 0, 0) with s+=|rs|^2, s-=|rp|^2. Non-diagonal
    entries capture phase differences (ellipsometry parameters); for
    lossless dielectrics rs, rp are real so U/V blocks are diagonal.
    The returned matrix acts when the ray's e1, e2 coincide with the
    s, p unit vectors -- the path basis builder guarantees that.
    """
    rs, rp, _, _, _ = _fresnel_amplitude(n1, n2, cos_i)
    ss = abs(rs) ** 2
    pp = abs(rp) ** 2
    sp = rs * np.conj(rp)
    re, im = sp.real, sp.imag
    a = (ss + pp) / 2.0
    b = (ss - pp) / 2.0
    return np.array([
        [a, b, 0.0, 0.0],
        [b, a, 0.0, 0.0],
        [0.0, 0.0,  re, im],
        [0.0, 0.0, -im, re],
    ])


def transmission(n1: complex, n2: complex, cos_i: float) -> np.ndarray:
    """Mueller matrix of refracted transmission, irradiance-normalized.

    Includes the (n2 cos_t)/(n1 cos_i) admittance factor so the matrix
    maps incident irradiance to transmitted irradiance; with it,
    R + T = I in the principal channels for lossless dielectrics.
    """
    rs, rp, cos_t, ts, tp = _fresnel_amplitude(n1, n2, cos_i)
    n1, n2 = complex(n1), complex(n2)
    # Irradiance ratio across the interface for lossless media.
    eta = ((n2 * cos_t) / (n1 * cos_i)).real
    ss = eta * abs(ts) ** 2
    pp = eta * abs(tp) ** 2
    sp = eta * ts * np.conj(tp)
    re, im = sp.real, sp.imag
    a = (ss + pp) / 2.0
    b = (ss - pp) / 2.0
    return np.array([
        [a, b, 0.0, 0.0],
        [b, a, 0.0, 0.0],
        [0.0, 0.0,  re, im],
        [0.0, 0.0, -im, re],
    ])


def reflectance_normal(n1: complex, n2: complex) -> float:
    rs, rp, *_ = _fresnel_amplitude(n1, n2, 1.0)
    return abs(rs) ** 2  # |rs|=|rp| at normal incidence


def brewster_angle(n1: float, n2: float) -> float:
    """Brewster incidence angle in radians (lossless dielectrics)."""
    return math.atan2(n2, n1)

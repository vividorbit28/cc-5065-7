"""Path-local transverse frames and their transport along rays.

A frame is (e1, e2, k): orthonormal, right-handed, e1 x e2 = k.
The s/p interface basis is built so that e1 = s and e2 = p x-direction,
which lets the (s,p)-form Mueller matrices act directly.

Near-normal degeneracy: when k is parallel to the world reference axis z,
the cross product that defines s vanishes. Two explicit policies are
implemented and tested against observable invariance:

* "transport": keep the previous frame's e1 (parallel-transport the basis);
* "reset":    re-anchor to the world x axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1.0e-12


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def initial_frame(k: np.ndarray, reference: np.ndarray | None = None) -> tuple:
    """Build the first transverse frame for a propagation direction k."""
    k = normalize(k)
    ref = np.array([0.0, 0.0, 1.0]) if reference is None else normalize(reference)
    if abs(np.dot(k, ref)) > 1.0 - 1.0e-9:
        # Degenerate: k parallel to reference axis. World x is a
        # reproducible fallback ("reset" anchor at the source).
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(k, ref)) > 1.0 - 1.0e-9:
            ref = np.array([0.0, 1.0, 0.0])
    e1 = normalize(np.cross(ref, k))
    e2 = normalize(np.cross(k, e1))
    return e1, e2, k


def reflect_vector(k: np.ndarray, normal: np.ndarray) -> np.ndarray:
    k = normalize(k)
    n = normalize(normal)
    return normalize(k - 2.0 * np.dot(k, n) * n)


def refract_vector(k, n1, n2, normal):
    """Refracted direction; returns None under total internal reflection."""
    k = normalize(k)
    n = normalize(normal)
    eta = complex(n1) / complex(n2)
    cos_i = -np.dot(k, n)
    sin2_t = (eta * eta * (1.0 - cos_i * cos_i))
    if isinstance(sin2_t, complex):
        sin2_t = sin2_t.real
    if sin2_t > 1.0:
        return None
    cos_t = math.sqrt(max(0.0, 1.0 - sin2_t))
    t = eta * k + (eta * cos_i - cos_t) * n
    return normalize(t.real if np.iscomplexobj(t) else t)


def interface_s_frame(k: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """s unit vector = normalize(normal x k); perpendicular to incidence plane."""
    s = np.cross(normal, k)
    norm = np.linalg.norm(s)
    if norm < 1.0e-10:
        return None  # normal-incidence degeneracy: no unique plane
    return s / norm


@dataclass
class TransportResult:
    frame: tuple                 # outgoing (e1, e2, k)
    angle_from_old_e1_deg: float # rotation of the new frame vs the old one
    degenerate: bool


def transport_reflect(previous_frame, normal, policy: str = "transport") -> TransportResult:
    """Propagate the transverse basis through a specular reflection.

    e1 is aligned with the interface s vector. At the degenerate
    near-normal point the s axis is undefined, so the policy selects
    parallel transport of the old e1 or a world-axis reset.
    """
    e1_old, e2_old, k_old = (normalize(x) for x in previous_frame)
    n = normalize(normal)
    k_new = reflect_vector(k_old, n)
    s = interface_s_frame(k_old, n)
    degenerate = s is None
    if degenerate:
        if policy == "transport":
            e1_new = e1_old - k_new * np.dot(e1_old, k_new)
            e1_new = normalize(e1_new)
        elif policy == "reset":
            anchor = np.array([1.0, 0.0, 0.0])
            e1_new = anchor - k_new * np.dot(anchor, k_new)
            if np.linalg.norm(e1_new) < 1.0e-10:
                anchor = np.array([0.0, 1.0, 0.0])
                e1_new = anchor - k_new * np.dot(anchor, k_new)
            e1_new = normalize(e1_new)
        else:
            raise ValueError(f"unknown degeneracy policy {policy!r}")
    else:
        e1_new = s
    e2_new = normalize(np.cross(k_new, e1_new))
    angle = frame_rotation_angle(previous_frame, (e1_new, e2_new, k_new))
    return TransportResult((e1_new, e2_new, k_new), angle, degenerate)


def transport_propagate(previous_frame, k_new) -> tuple:
    """Parallel-transport a frame to a new direction (free flight)."""
    e1_old, _, k_old = (normalize(x) for x in previous_frame)
    k_new = normalize(k_new)
    e1_new = e1_old - k_new * np.dot(e1_old, k_new)
    norm = np.linalg.norm(e1_new)
    if norm < 1.0e-10:
        raise ValueError("free-flight transport folded onto the axis")
    e1_new = e1_new / norm
    e2_new = normalize(np.cross(k_new, e1_new))
    return e1_new, e2_new, k_new


def frame_rotation_angle(frame_a, frame_b) -> float:
    """Signed angle [deg] rotating frame_a's basis into frame_b's.

    Both frames share (nearly) the same k. The angle follows from the
    (e1, e2) components of frame_b's e1 expressed in frame_a:
    e1_b = cos a * e1_a + sin a * e2_a.
    """
    e1a, e2a, _ = frame_a
    e1b, _, _ = frame_b
    c = float(np.clip(np.dot(e1b, e1a), -1.0, 1.0))
    s = float(np.clip(np.dot(e1b, e2a), -1.0, 1.0))
    return math.degrees(math.atan2(s, c))


def lens_frame(ray_frame, up_hint=None) -> tuple:
    """Camera/lens coordinates for an orthographic sensor.

    The sensor looks along -z_cam with horizontal x_cam, vertical y_cam;
    the incoming ray propagates toward the sensor, k ~ +z_cam.
    Returns the lens frame (e1_lens=x_cam, e2_lens=y_cam, k_lens=z_cam)
    and the rotation angle [deg] from the ray basis to the lens basis.
    """
    _, _, k = ray_frame
    z_cam = normalize(k)
    hint = np.array([0.0, 1.0, 0.0]) if up_hint is None else normalize(up_hint)
    x_cam = normalize(np.cross(hint, z_cam))
    y_cam = normalize(np.cross(z_cam, x_cam))
    lens = (x_cam, y_cam, z_cam)
    return lens, frame_rotation_angle(ray_frame, lens)

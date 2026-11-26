"""Analytic plane scenes and the polarized renderer.

Two scenes are provided:

* :class:`CorrugatedPlane` -- one specular reflection per pixel. Incidence
  angle spans continuously across the frame, including a near-normal
  column where the s/p reference axis degenerates.
* :class:`VGroove` -- two sequential reflections inside a folded groove,
  exercising the multi-bounce frame transport and the degeneracy policy.

The renderer returns per-pixel Stokes states (in lens coordinates) plus
intensity maps for any set of analyzer angles. No pixel is ever turned
into a color without passing the Stokes feasibility check first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import (
    frame_rotation_angle, initial_frame, normalize, reflect_vector,
    transport_reflect,
)
from .mueller import reflection, rotation
from .path import PolarizedPath, PathEngine
from .stokes import check_feasible, zero


@dataclass
class RenderResult:
    stokes_grid: np.ndarray      # (H, W, 4) in lens frame
    analyzer_angles: np.ndarray  # (A,) degrees
    intensity: np.ndarray        # (A, H, W)
    incidence_deg: np.ndarray    # (H, W)
    valid: np.ndarray            # (H, W) bool, ray contributed
    diagnostics: list
    engine: PathEngine

    def total_polarized_power(self) -> np.ndarray:
        return np.sqrt(
            self.stokes_grid[..., 1] ** 2
            + self.stokes_grid[..., 2] ** 2
            + self.stokes_grid[..., 3] ** 2
        )

    def dop_map(self) -> np.ndarray:
        i = self.stokes_grid[..., 0]
        out = np.zeros_like(i)
        nz = i > 0
        out[nz] = self.total_polarized_power()[nz] / i[nz]
        return out


class CorrugatedPlane:
    """Plane z = A sin(2 pi x / period): the surface normal varies with x.

    Under orthographic viewing along z, the reflected incidence angle is
    determined per column by the local normal, so the rendered image spans
    a continuous range of incidence angles and crosses the near-normal
    degeneracy where the normal is parallel to the view axis.
    """

    def __init__(self, amplitude: float = 0.35, period: float = 2.0,
                 n1=1.0, n2=1.5):
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.n1, self.n2 = complex(n1), complex(n2)

    def normal_at(self, x: float) -> np.ndarray:
        # Surface f = A sin(kx) - z = 0 ; grad f = (A k cos(kx), 0, -1).
        k = 2.0 * np.pi / self.period
        return normalize(np.array([self.amplitude * k * np.cos(k * x),
                                   0.0, -1.0]))


class VGroove:
    """Symmetric V-groove in the x-z plane, opening toward +z.

    Vertex at (0, 0, -depth). Left facet normal points +x, right facet -x.
    Rays enter from +z travelling toward -z and undergo two reflections
    before returning toward the camera.
    """

    def __init__(self, half_angle_deg: float = 45.0, depth: float = 1.0,
                 n1=1.0, n2=1.5):
        self.alpha = np.radians(half_angle_deg)
        self.depth = float(depth)
        self.n1, self.n2 = complex(n1), complex(n2)
        self.n_left = normalize(np.array([np.cos(self.alpha), 0.0,
                                          np.sin(self.alpha)]))
        self.n_right = normalize(np.array([-np.cos(self.alpha), 0.0,
                                           np.sin(self.alpha)]))
        # Half width at the opening z=0.
        self.half_width = depth / np.tan(self.alpha)


def _lens_coordinates(frame):
    """Ray -> lens basis transform for the orthographic camera.

    Lens basis (x_cam, y_cam, k) with x_cam=+x, y_cam=+y, k the outgoing
    direction. The state is re-expressed with the exact passive rotation
    ``frame_rotation_angle(frame, lens)`` so that a physical analyzer
    fixed in the lens frame reads ``analyzer(theta)`` on ``R(a) S``. Using
    the negated angle instead mirrors the U channel and is invisible for
    Q-only states but wrong for any U!=0 state.
    """
    e1, e2, k = frame
    x_cam = np.array([1.0, 0.0, 0.0])
    y_cam = np.array([0.0, 1.0, 0.0])
    k = normalize(k)
    lens = (x_cam, y_cam, k)
    angle = frame_rotation_angle(frame, lens)
    return lens, angle


class PolarizedRenderer:
    def __init__(self, degeneracy_policy: str = "transport"):
        self.engine = PathEngine(degeneracy_policy)

    # -- single reflection -------------------------------------------------

    def render_corrugated_plane(self, scene: "CorrugatedPlane",
                                source_stokes: np.ndarray,
                                width: int = 320, height: int = 240,
                                extent: float = 1.0,
                                analyzer_angles=(0.0, 45.0, 90.0, 135.0),
                                extinction: float = 0.0) -> RenderResult:
        source_stokes = np.asarray(source_stokes, dtype=float)
        check_feasible(source_stokes, raise_on_fail=True)
        xs = np.linspace(-extent, extent, width)
        ys = np.linspace(-extent, extent, height)
        grid = np.zeros((height, width, 4))
        inc = np.zeros((height, width))
        valid = np.zeros((height, width), dtype=bool)
        diagnostics = []
        k_out0 = np.array([0.0, 0.0, 1.0])
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                n = scene.normal_at(x)
                k_in = reflect_vector(k_out0, n)
                cos_i = abs(float(np.dot(-k_in, n)))
                theta = np.degrees(np.arccos(np.clip(cos_i, -1, 1)))
                inc[j, i] = theta
                frame0 = initial_frame(k_in)
                path = PolarizedPath(np.array(source_stokes, dtype=float), frame0)
                try:
                    self.engine.reflect(path, n, scene.n1, scene.n2)
                    lens, ang = _lens_coordinates(path.frame)
                    s_lens = rotation(ang) @ path.stokes
                    s_lens = _guard(s_lens, diagnostics, (j, i))
                    grid[j, i] = s_lens
                    valid[j, i] = True
                except Exception as exc:  # pragma: no cover - defensive
                    diagnostics.append(f"pixel({j},{i}): {exc}")
                    grid[j, i] = zero()
        return self._finalize(grid, inc, valid, diagnostics, analyzer_angles,
                              extinction)

    # -- double reflection -------------------------------------------------

    def render_v_groove(self, scene: VGroove, source_stokes: np.ndarray,
                        width: int = 320, height: int = 240,
                        analyzer_angles=(0.0, 45.0, 90.0, 135.0),
                        extinction: float = 0.0) -> RenderResult:
        source_stokes = np.asarray(source_stokes, dtype=float)
        check_feasible(source_stokes, raise_on_fail=True)
        xs = np.linspace(-scene.half_width, scene.half_width, width)
        ys = np.linspace(-scene.half_width, scene.half_width, height)
        grid = np.zeros((height, width, 4))
        inc = np.zeros((height, width))
        valid = np.zeros((height, width), dtype=bool)
        diagnostics = []
        k_in = np.array([0.0, 0.0, -1.0])
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if abs(x) < 1e-12:
                    continue  # exact vertex: undefined two-face ordering
                n1v = scene.n_left if x > 0 else scene.n_right
                n2v = scene.n_right if x > 0 else scene.n_left
                frame0 = initial_frame(k_in)
                path = PolarizedPath(np.array(source_stokes, dtype=float), frame0)
                try:
                    self.engine.reflect(path, n1v, scene.n1, scene.n2)
                    k_mid = path.frame[2]
                    # Geometry check: the second facet is reached before exit.
                    if np.dot(k_mid, n2v) >= 0:
                        continue
                    self.engine.reflect(path, n2v, scene.n1, scene.n2)
                    if abs(path.frame[2][2] - 1.0) > 1e-9:
                        continue  # double reflection must return along +z
                    lens, ang = _lens_coordinates(path.frame)
                    s_lens = rotation(ang) @ path.stokes
                    s_lens = _guard(s_lens, diagnostics, (j, i))
                    grid[j, i] = s_lens
                    valid[j, i] = True
                    cos_i = abs(float(np.dot(-k_in, n1v)))
                    inc[j, i] = np.degrees(np.arccos(np.clip(cos_i, -1, 1)))
                except Exception as exc:  # pragma: no cover - defensive
                    diagnostics.append(f"pixel({j},{i}): {exc}")
                    grid[j, i] = zero()
        return self._finalize(grid, inc, valid, diagnostics, analyzer_angles,
                              extinction)

    # -- analyzer integration ---------------------------------------------

    def _finalize(self, grid, inc, valid, diagnostics, analyzer_angles,
                  extinction) -> RenderResult:
        angles = np.asarray(analyzer_angles, dtype=float)
        a_count = len(angles)
        h, w, _ = grid.shape
        intensity = np.zeros((a_count, h, w))
        for a, ang in enumerate(angles):
            m = np.zeros((h, w))
            for j in range(h):
                for i in range(w):
                    if not valid[j, i]:
                        continue
                    s = grid[j, i]
                    mp = linear_polarizer_matrix(ang, extinction)
                    out = mp @ s
                    out = _guard(out, diagnostics, (j, i, "analyzer"))
                    m[j, i] = out[0]
            intensity[a] = m
        return RenderResult(grid, angles, intensity, inc, valid,
                            diagnostics, self.engine)


def linear_polarizer_matrix(theta_deg: float, extinction: float = 0.0):
    from .mueller import linear_polarizer
    return linear_polarizer(theta_deg, extinction)


def _guard(s, diagnostics, where):
    """Every state entering output/color conversion passes the cone check."""
    report = check_feasible(s)
    if not report.ok:
        diagnostics.append(f"state at {where}: {report.summary()}")
        raise ValueError(report.summary())
    return s

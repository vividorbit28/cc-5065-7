"""Polarized surface rendering prototype: Stokes/Mueller path propagation."""

from .stokes import (
    FeasibilityReport,
    UnphysicalStokesError,
    check_feasible,
    degrees_of_polarization,
    from_components,
    linear,
    sanitize,
    zero,
)
from .mueller import (
    analyzer_intensity,
    brewster_angle,
    linear_polarizer,
    reflection,
    reflectance_normal,
    rotation,
    transmission,
)
from .frames import initial_frame, lens_frame, transport_reflect
from .path import PathEngine, PolarizedPath, no_light_path
from .scene import PolarizedRenderer, CorrugatedPlane, VGroove, RenderResult
from .diagnostics import (
    classify_analyzer,
    compare_degeneracy_policies,
    diagnose,
    format_report,
    reverse_readout_check,
    run_recipe,
)

__all__ = [
    "FeasibilityReport",
    "UnphysicalStokesError",
    "check_feasible",
    "degrees_of_polarization",
    "from_components",
    "linear",
    "sanitize",
    "zero",
    "analyzer_intensity",
    "brewster_angle",
    "linear_polarizer",
    "reflection",
    "reflectance_normal",
    "rotation",
    "transmission",
    "initial_frame",
    "lens_frame",
    "transport_reflect",
    "PathEngine",
    "PolarizedPath",
    "no_light_path",
    "PolarizedRenderer",
    "CorrugatedPlane",
    "VGroove",
    "RenderResult",
    "diagnose",
    "format_report",
    "classify_analyzer",
    "reverse_readout_check",
    "compare_degeneracy_policies",
    "run_recipe",
]

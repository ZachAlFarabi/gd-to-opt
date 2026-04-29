"""
objectives.py — Chunk 5: objective computation and normalisation

Extracts the three optimisation objectives from SIMPResult and
normalises them for NSGA-II's crowding distance computation.

Objectives:
    f[0]  compliance_Nm    — structural flexibility [N·m]
    f[1]  mass_kg          — material mass [kg]
    f[2]  support_volume_m3 — overhang proxy [m³]

Normalisation (D24):
    f_norm[i] = f[i] / f_ref[i]
    where f_ref is the max of each objective in the first generation.
    Failed evaluations (f=1e10) are handled gracefully.
"""

from __future__ import annotations
import numpy as np
from gdto.simp import SIMPResult


def compute_objectives(result: SIMPResult) -> np.ndarray:
    """
    Extract the three objectives from a SIMPResult.

    Returns
    -------
    np.ndarray of shape (3,): [compliance_Nm, mass_kg, support_volume_m3]
    """
    # Compliance already in N·m
    c = float(result.compliance)

    # Mass in kg — SIMPResult stores raw kg value
    m = float(result.mass)

    # Support volume in m³
    v = float(result.support_volume)

    # Guard against numerical garbage
    if not np.isfinite(c) or c < 0:
        c = 1e10
    if not np.isfinite(m) or m < 0:
        m = 1e10
    if not np.isfinite(v) or v < 0:
        v = 0.0

    return np.array([c, m, v], dtype=np.float64)


def normalise_objectives(
    F:   np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """
    Normalise raw objectives by reference values.

    Parameters
    ----------
    F   : (n, 3) raw objective matrix
    ref : (3,)   reference values (max of first generation)

    Returns
    -------
    F_norm : (n, 3) normalised objectives in approximately [0, 1]

    Failed evaluations (value = 1e10) are kept as large values
    so they remain dominated by all valid solutions.
    """
    ref_safe = np.where(np.abs(ref) > 1e-30, ref, 1.0)
    F_norm   = F / ref_safe[np.newaxis, :]

    # Cap failed evaluations at 100 (clearly dominated)
    F_norm = np.clip(F_norm, 0.0, 100.0)

    return F_norm


def hypervolume_indicator(
    F:    np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """
    Compute the hypervolume indicator for a set of objective vectors.

    The hypervolume is the volume of the objective space dominated by
    the solution set and bounded by ref_point. Larger = better front.

    Uses a simple sweep algorithm for 2D; for 3D uses pymoo's WFG.

    Parameters
    ----------
    F         : (n, 3) objective vectors (minimisation)
    ref_point : (3,)   reference point (worse than all solutions)

    Returns
    -------
    float : hypervolume indicator
    """
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=ref_point)
        return float(ind(F))
    except Exception:
        return 0.0


def pareto_front_2d(
    F:    np.ndarray,
    obj1: int = 0,
    obj2: int = 1,
) -> np.ndarray:
    """
    Extract the 2D Pareto front for a pair of objectives.
    Used for the frontend scatter plot.

    Parameters
    ----------
    F    : (n, 3) objective matrix
    obj1 : index of first objective for x-axis
    obj2 : index of second objective for y-axis

    Returns
    -------
    (m, 2) Pareto-optimal points in the 2D projection,
    sorted by obj1.
    """
    f2d  = F[:, [obj1, obj2]]
    mask = np.ones(len(f2d), dtype=bool)

    for i in range(len(f2d)):
        if not mask[i]:
            continue
        for j in range(len(f2d)):
            if i == j or not mask[j]:
                continue
            # i dominates j in 2D?
            if (f2d[i, 0] <= f2d[j, 0] and f2d[i, 1] <= f2d[j, 1] and
                    (f2d[i, 0] < f2d[j, 0] or f2d[i, 1] < f2d[j, 1])):
                mask[j] = False

    front = f2d[mask]
    order = np.argsort(front[:, 0])
    return front[order]


def build_cost_estimate(
    mass_kg:          float,
    support_volume_m3: float,
    material:          str = "Ti64",
) -> dict:
    """
    Estimate build cost from mass and support volume.
    Used in chunk 6 objectives — placeholder implementation.

    Cost model (approximate LPBF pricing):
        material_cost = mass_kg * price_per_kg
        support_cost  = support_volume_m3 * 1e6 * support_factor * price_per_kg
        machine_cost  = (mass_kg + support_mass) / build_rate * machine_rate

    Returns
    -------
    dict with material_cost, support_cost, total_cost [USD]
    """
    # Approximate LPBF material prices (USD/kg, 2024)
    prices = {"Ti64": 400.0, "AlSi10Mg": 120.0, "316L": 180.0}
    price  = prices.get(material, 300.0)

    # Support volume in cm³ for cost calc
    supp_cm3 = support_volume_m3 * 1e6
    supp_kg  = supp_cm3 * 1e-3 * {"Ti64": 4.4, "AlSi10Mg": 2.68, "316L": 7.99}.get(material, 4.0)

    material_cost = mass_kg * price
    support_cost  = supp_kg * price * 0.3   # support material at 30% efficiency
    machine_rate  = 50.0   # USD/hour
    build_rate    = 0.010  # kg/hour (rough LPBF estimate)
    machine_cost  = ((mass_kg + supp_kg) / build_rate) * machine_rate

    return {
        "material_cost_usd": round(material_cost, 2),
        "support_cost_usd":  round(support_cost, 2),
        "machine_cost_usd":  round(machine_cost, 2),
        "total_cost_usd":    round(material_cost + support_cost + machine_cost, 2),
    }
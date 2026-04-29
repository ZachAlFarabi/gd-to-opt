"""
tests/test_gd_loop.py — Chunk 5: GD loop tests

Tests:
    1.  compute_objectives returns shape (3,)
    2.  compute_objectives values are non-negative
    3.  normalise_objectives divides by reference
    4.  normalise_objectives clips failed evaluations
    5.  normalise_objectives handles zero reference safely
    6.  pareto_front_2d returns non-dominated points
    7.  pareto_front_2d is sorted by first objective
    8.  hypervolume_indicator returns positive float
    9.  GDConfig defaults are sensible
    10. GDConfig n_workers >= 1
    11. ParetoSolution to_dict excludes stl_b64
    12. TopOptProblem constructs with correct dimensions
    13. TopOptProblem variable bounds correct
    14. _evaluate_individual runs on tiny mesh
    15. GDResult to_dict has required keys
    16. run_gd produces GDResult (tiny config, 1 gen, pop=4)
    17. Pareto front solutions are non-dominated
    18. All Pareto objectives are finite and positive
    19. build_cost_estimate returns positive costs
    20. GDResult all_objectives shape is (n_eval, 3)
"""

from __future__ import annotations

import numpy as np
import pytest

from gdto.objectives import (
    compute_objectives, normalise_objectives,
    pareto_front_2d, hypervolume_indicator, build_cost_estimate,
)
from gdto.gd_loop import (
    GDConfig, GDResult, ParetoSolution, TopOptProblem,
    _evaluate_individual, run_gd, XL, XU,
)
from gdto.simp import SIMPResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_simp_result():
    """Minimal SIMPResult for objective extraction tests."""
    return SIMPResult(
        compliance         = 4.3e-3,
        mass               = 0.847,
        support_volume     = 1.2e-4,
        rho                = np.full(512, 0.4, dtype=np.float32),
        converged          = True,
        n_iterations       = 20,
        compliance_history = np.linspace(8e-3, 4.3e-3, 20),
        change_history     = np.linspace(0.2, 0.01, 20),
        bisection_iters_mean = 15.0,
        mu_history         = np.ones(20) * 1e-6,
        scheme_used        = "OC",
        wall_time_s        = 12.3,
        design_params      = {"volume_fraction": 0.4},
        density_snapshots  = None,
        snapshot_iters     = None,
        T_field_final      = None,
    )

@pytest.fixture(scope="module")
def tiny_gd_config():
    """GDConfig for fast testing: 8³ mesh, 1 gen, pop=4."""
    return GDConfig(
        pop_size    = 4,
        n_gen       = 1,
        n_workers   = 1,
        simp_iter   = 5,
        simp_tol    = 1.0,    # very loose — terminates immediately
        mesh_nx     = 8,
        mesh_ny     = 8,
        mesh_nz     = 8,
        material    = "Ti64",
        fixed_faces = ["zmin"],
        load_face   = "zmax",
        load_dir    = 2,
        load_mag    = -1.0,
        lx          = 1.0, ly=1.0, lz=1.0,
    )


# ── compute_objectives ────────────────────────────────────────────────────────

def test_compute_objectives_shape(dummy_simp_result):
    objs = compute_objectives(dummy_simp_result)
    assert objs.shape == (3,)

def test_compute_objectives_nonneg(dummy_simp_result):
    objs = compute_objectives(dummy_simp_result)
    assert np.all(objs >= 0)

def test_compute_objectives_values(dummy_simp_result):
    objs = compute_objectives(dummy_simp_result)
    np.testing.assert_allclose(objs[0], 4.3e-3, rtol=1e-6)
    np.testing.assert_allclose(objs[1], 0.847,  rtol=1e-6)
    np.testing.assert_allclose(objs[2], 1.2e-4, rtol=1e-6)

def test_compute_objectives_nan_compliance():
    """NaN compliance should be replaced with 1e10."""
    result = SIMPResult(
        compliance=float('nan'), mass=0.5, support_volume=0.0,
        rho=np.zeros(8), converged=False, n_iterations=0,
        compliance_history=np.array([]), change_history=np.array([]),
        bisection_iters_mean=0, mu_history=np.array([]),
        scheme_used='OC', wall_time_s=0, design_params={},
        density_snapshots=None, snapshot_iters=None, T_field_final=None,
    )
    objs = compute_objectives(result)
    assert objs[0] == 1e10


# ── normalise_objectives ──────────────────────────────────────────────────────

def test_normalise_shape():
    F   = np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
    ref = np.array([2.0, 4.0, 6.0])
    F_n = normalise_objectives(F, ref)
    assert F_n.shape == (2, 3)

def test_normalise_values():
    F   = np.array([[2.0, 4.0, 6.0]])
    ref = np.array([2.0, 4.0, 6.0])
    F_n = normalise_objectives(F, ref)
    np.testing.assert_allclose(F_n, [[1.0, 1.0, 1.0]])

def test_normalise_clips_failed():
    F   = np.array([[1e10, 1e10, 1e10]])
    ref = np.array([1e-3, 1.0, 1e-4])
    F_n = normalise_objectives(F, ref)
    assert np.all(F_n <= 100.0)

def test_normalise_zero_reference():
    F   = np.array([[1.0, 2.0, 3.0]])
    ref = np.array([0.0, 0.0, 0.0])
    F_n = normalise_objectives(F, ref)   # should not raise
    assert np.all(np.isfinite(F_n))


# ── pareto_front_2d ──────────────────────────────────────────────────────────

def test_pareto_2d_nondominated():
    # Point (1,3) is dominated by (1,2) and (0,2)
    F = np.array([[0.0,2.0,0],[1.0,2.0,0],[1.0,3.0,0],[2.0,1.0,0]])
    front = pareto_front_2d(F, 0, 1)
    # Dominated point (1,3) should not appear
    assert not any(np.allclose(row, [1.0,3.0]) for row in front)

def test_pareto_2d_sorted():
    F = np.array([[3.0,1.0,0],[1.0,3.0,0],[2.0,2.0,0]])
    front = pareto_front_2d(F, 0, 1)
    assert front[0,0] <= front[-1,0]

def test_pareto_2d_shape():
    F = np.array([[1.0,3.0,0],[2.0,2.0,0],[3.0,1.0,0]])
    front = pareto_front_2d(F, 0, 1)
    assert front.shape[1] == 2


# ── hypervolume_indicator ─────────────────────────────────────────────────────

def test_hv_positive():
    F   = np.array([[0.2, 0.5, 0.3],[0.4, 0.2, 0.4]])
    ref = np.array([1.0, 1.0, 1.0])
    hv  = hypervolume_indicator(F, ref)
    assert hv > 0


# ── GDConfig ─────────────────────────────────────────────────────────────────

def test_gd_config_defaults():
    cfg = GDConfig()
    assert cfg.pop_size == 40
    assert cfg.n_gen    == 5
    assert cfg.material == "Ti64"

def test_gd_config_n_workers_positive():
    cfg = GDConfig()
    assert cfg.n_workers >= 1


# ── ParetoSolution ────────────────────────────────────────────────────────────

def test_pareto_solution_to_dict_no_stl():
    s = ParetoSolution(
        solution_id=0, volume_fraction=0.4, overhang_weight=0.0,
        build_orientation_deg=0.0, min_feature_size=1.5,
        compliance_Nm=4.3e-3, mass_kg=0.847, support_volume_m3=1.2e-4,
        stl_b64="fakestldata",
    )
    d = s.to_dict()
    assert "stl_b64" not in d
    assert "compliance_Nm" in d


# ── TopOptProblem ─────────────────────────────────────────────────────────────

def test_problem_dimensions(tiny_gd_config):
    prob = TopOptProblem(tiny_gd_config)
    assert prob.n_var == 4
    assert prob.n_obj == 3
    assert prob.n_constr == 0

def test_problem_bounds(tiny_gd_config):
    prob = TopOptProblem(tiny_gd_config)
    np.testing.assert_array_equal(prob.xl, XL)
    np.testing.assert_array_equal(prob.xu, XU)


# ── _evaluate_individual ─────────────────────────────────────────────────────

def test_evaluate_individual_runs(tiny_gd_config):
    """Single evaluation on tiny mesh must return finite objectives."""
    x       = np.array([0.4, 0.0, 0.0, 1.5])
    cfg_dict = tiny_gd_config.__dict__
    c, m, v, err = _evaluate_individual((x, cfg_dict))
    assert err == "", f"Evaluation failed: {err[:300]}"
    assert c > 0 and np.isfinite(c)
    assert m > 0 and np.isfinite(m)
    assert v >= 0 and np.isfinite(v)


# ── run_gd ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gd_result(tiny_gd_config):
    return run_gd(tiny_gd_config)

def test_run_gd_returns_result(gd_result):
    assert isinstance(gd_result, GDResult)

def test_run_gd_has_solutions(gd_result):
    assert len(gd_result.pareto_solutions) > 0

def test_run_gd_objectives_finite(gd_result):
    for s in gd_result.pareto_solutions:
        assert np.isfinite(s.compliance_Nm)
        assert np.isfinite(s.mass_kg)
        assert np.isfinite(s.support_volume_m3)

def test_run_gd_objectives_positive(gd_result):
    for s in gd_result.pareto_solutions:
        assert s.compliance_Nm > 0
        assert s.mass_kg > 0
        assert s.support_volume_m3 >= 0

def test_run_gd_pareto_nondominated(gd_result):
    """No Pareto solution should be dominated by another."""
    sols = gd_result.pareto_solutions
    for i, si in enumerate(sols):
        for j, sj in enumerate(sols):
            if i == j:
                continue
            fi = np.array([si.compliance_Nm, si.mass_kg, si.support_volume_m3])
            fj = np.array([sj.compliance_Nm, sj.mass_kg, sj.support_volume_m3])
            # sj should NOT dominate si
            assert not (np.all(fj <= fi) and np.any(fj < fi)), \
                f"Solution {j} dominates solution {i}"

def test_run_gd_result_to_dict(gd_result):
    d = gd_result.to_dict()
    assert "n_solutions"   in d
    assert "n_generations" in d
    assert "solutions"     in d
    assert "wall_time_s"   in d

def test_run_gd_all_objectives_shape(gd_result, tiny_gd_config):
    n_eval = gd_result.n_evaluations
    assert gd_result.all_objectives.shape    == (n_eval, 3)
    assert gd_result.all_design_params.shape == (n_eval, 4)


# ── build_cost_estimate ───────────────────────────────────────────────────────

def test_build_cost_positive():
    cost = build_cost_estimate(0.5, 1e-5, "Ti64")
    assert cost["total_cost_usd"] > 0
    assert cost["material_cost_usd"] > 0
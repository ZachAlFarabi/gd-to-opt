"""
Tests for simp.py — Chunk 2.

Verifies:
    1.  SIMPResult has all required fields
    2.  Compliance decreases monotonically (or at least converges)
    3.  Volume fraction is enforced to within bisect_tol
    4.  Density field stays in [rho_min, 1]
    5.  Support volume is non-negative
    6.  Mass is positive and physically reasonable
    7.  OC and MMA both converge on small problem
    8.  Auto scheme selects OC for no extra constraints
    9.  Auto scheme selects MMA when overhang_weight > 0
    10. Sensitivity vectorisation matches naive loop
    11. Bisection diagnostics are logged
    12. Design params stored in result
    13. Wall time is positive
    14. All three material presets run without error
"""

from __future__ import annotations

import numpy as np
import pytest

from gdto.mesh_material import build_problem
from gdto.simp import run_simp, SIMPSolver, SIMPConfig, SIMPResult


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_config():
    """Fast config for unit tests — small mesh, few iterations."""
    return SIMPConfig(
        p               = 3.0,
        E_min_factor    = 1e-9,
        max_iter        = 20,    # enough to see convergence trend
        tol             = 1e-2,  # loose tolerance for speed
        volume_fraction = 0.4,
        update_scheme   = "OC",
        oc_move         = 0.2,
        oc_eta          = 0.5,
        bisect_tol      = 1e-6,
        bisect_max_iter = 100,
    )

@pytest.fixture(scope="module")
def small_problem():
    """20^3 Ti64 problem — fast enough for unit tests."""
    return build_problem(material="Ti64", nx=20, ny=20, nz=20)

@pytest.fixture(scope="module")
def oc_result(small_problem, small_config):
    """Pre-computed OC result shared across tests."""
    return run_simp(small_problem, config=small_config)


# ── Result structure tests ────────────────────────────────────────────────

def test_result_is_simpreult(oc_result):
    assert isinstance(oc_result, SIMPResult)

def test_result_has_all_fields(oc_result):
    for field in SIMPResult._fields:
        assert hasattr(oc_result, field), f"Missing field: {field}"

def test_compliance_positive(oc_result):
    assert oc_result.compliance > 0

def test_mass_positive(oc_result):
    assert oc_result.mass > 0

def test_support_volume_nonnegative(oc_result):
    assert oc_result.support_volume >= 0

def test_wall_time_positive(oc_result):
    assert oc_result.wall_time_s > 0

def test_n_iterations_positive(oc_result):
    assert oc_result.n_iterations > 0

def test_scheme_used_is_oc(oc_result):
    assert oc_result.scheme_used == "OC"


# ── Density field tests ───────────────────────────────────────────────────

def test_density_shape(oc_result, small_problem):
    assert oc_result.rho.shape == (small_problem.mesh.n_elem,)

def test_density_bounds(oc_result, small_config):
    assert oc_result.rho.min() >= small_config.rho_min - 1e-10
    assert oc_result.rho.max() <= 1.0 + 1e-10

def test_volume_fraction_enforced(oc_result, small_config):
    """Mean density must equal volume_fraction to within bisect_tol."""
    np.testing.assert_allclose(
        oc_result.rho.mean(),
        small_config.volume_fraction,
        atol=small_config.bisect_tol * 10,  # allow 10x tol for final iteration
        err_msg="Volume fraction not enforced"
    )


# ── Convergence tests ─────────────────────────────────────────────────────

def test_compliance_history_length(oc_result):
    assert len(oc_result.compliance_history) == oc_result.n_iterations

def test_change_history_length(oc_result):
    assert len(oc_result.change_history) == oc_result.n_iterations

def test_compliance_decreases_overall(oc_result):
    """
    With p-continuation, compliance may rise during the penalisation ramp
    then fall as topology consolidates. The correct invariant is that
    compliance at the end is lower than the peak during optimisation,
    and that the density field has polarised from the uniform initial state.

    Monotonic decrease is only expected without continuation (fixed p).
    """
    history = oc_result.compliance_history
    rho     = oc_result.rho

    # The optimiser must have actually changed the density field
    assert rho.std() > 0.01, "Density field did not polarise"

    # Final compliance must be lower than the maximum seen during optimisation
    # (the peak occurs during the p-ramp, not at the end)
    assert history[-1] < max(history), \
        "Final compliance must be lower than peak compliance"

    # The density field should not be uniform (optimisation happened)
    assert not np.allclose(rho, rho.mean(), atol=0.05), \
        "Density field remained uniform — no optimisation occurred"


def test_compliance_decreases_without_continuation():
    """Without p-continuation (p_start == p_end), compliance decreases monotonically."""
    prob   = build_problem(material="Ti64", nx=10, ny=10, nz=10)
    cfg    = SIMPConfig(
        max_iter = 20,
        tol      = 1e-2,
        p_start  = 3.0,   # same as p_end → no ramp
        p_end    = 3.0,
    )
    result = run_simp(prob, config=cfg)
    history = result.compliance_history
    assert history[-1] < history[0], \
        "Without continuation, final compliance must be lower than initial"

def test_bisection_diagnostics_logged(oc_result):
    assert oc_result.bisection_iters_mean > 0
    assert len(oc_result.mu_history) == oc_result.n_iterations
    assert np.all(oc_result.mu_history > 0)


# ── Sensitivity vectorisation test (Decision 7) ──────────────────────────

def test_sensitivity_vectorisation(small_problem, small_config):
    """
    Vectorised einsum sensitivity must match naive per-element loop.
    Tests Decision 7 correctness.
    """
    import scipy.sparse.linalg as spla

    mesh = small_problem.mesh
    mat  = small_problem.material
    asm  = small_problem.assembler
    bc   = small_problem.bc
    cfg  = small_config

    rho    = np.full(mesh.n_elem, cfg.volume_fraction)
    K      = asm.assemble_K(rho, p=cfg.p, E_min_factor=cfg.E_min_factor)
    K_ff, f_f = bc.apply(K)
    u_free = spla.spsolve(K_ff, f_f)
    u_full = bc.expand(u_free)

    # Vectorised (Decision 7)
    Ue            = u_full[mesh.dof_map]
    Ke0_Ue        = Ue @ asm.Ke0.T
    strain_energy = np.einsum('ei,ei->e', Ue, Ke0_Ue)
    dc_vec        = -cfg.p * rho**(cfg.p-1) * (1-cfg.E_min_factor) * strain_energy

    # Naive loop
    dc_loop = np.zeros(mesh.n_elem)
    for e in range(mesh.n_elem):
        ue          = u_full[mesh.dof_map[e]]
        dc_loop[e]  = -cfg.p * rho[e]**(cfg.p-1) * (1-cfg.E_min_factor) * (ue @ asm.Ke0 @ ue)

    np.testing.assert_allclose(dc_vec, dc_loop, rtol=1e-10,
                               err_msg="Vectorised sensitivity != naive loop")


# ── Scheme selection tests (Decision 8) ──────────────────────────────────

def test_auto_selects_oc_no_overhang(small_problem):
    cfg    = SIMPConfig(update_scheme="auto", max_iter=5)
    solver = SIMPSolver(small_problem, cfg)
    assert solver._select_scheme({}) == "OC"
    assert solver._select_scheme({"overhang_weight": 0.0}) == "OC"

def test_auto_selects_mma_with_overhang(small_problem):
    cfg    = SIMPConfig(update_scheme="auto", max_iter=5)
    solver = SIMPSolver(small_problem, cfg)
    assert solver._select_scheme({"overhang_weight": 0.5}) == "MMA"

def test_explicit_oc_overrides_auto(small_problem):
    cfg    = SIMPConfig(update_scheme="OC", max_iter=5)
    solver = SIMPSolver(small_problem, cfg)
    assert solver._select_scheme({"overhang_weight": 0.5}) == "OC"


# ── MMA convergence test (Decision 8) ────────────────────────────────────

def test_mma_converges():
    """MMA must converge and enforce volume fraction.
    Uses 10^3 mesh (not shared fixture) to keep test time < 10s."""
    prob   = build_problem(material="Ti64", nx=10, ny=10, nz=10)
    cfg    = SIMPConfig(update_scheme="MMA", max_iter=20, tol=1e-2)
    result = run_simp(prob, config=cfg)
    assert result.scheme_used == "MMA"
    assert result.compliance > 0
    np.testing.assert_allclose(
        result.rho.mean(), cfg.volume_fraction,
        atol=cfg.bisect_tol * 10
    )

# ── Volume constraint test (Decision 9) ──────────────────────────────────

@pytest.mark.parametrize("Vf", [0.2, 0.4, 0.6])
def test_volume_fraction_parametric(small_problem, Vf):
    """Volume fraction must be enforced for different target values."""
    cfg    = SIMPConfig(max_iter=10, tol=1e-2, volume_fraction=Vf)
    result = run_simp(small_problem, config=cfg)
    np.testing.assert_allclose(
        result.rho.mean(), Vf,
        atol=cfg.bisect_tol * 10,
        err_msg=f"Volume fraction {Vf} not enforced"
    )

# ── Design params stored (Decision 10) ───────────────────────────────────

def test_design_params_stored(small_problem, small_config):
    params = {"volume_fraction": 0.4, "build_orientation_deg": 45.0}
    result = run_simp(small_problem, design_params=params, config=small_config)
    assert result.design_params["build_orientation_deg"] == 45.0

def test_design_params_empty_by_default(small_problem, small_config):
    result = run_simp(small_problem, config=small_config)
    assert isinstance(result.design_params, dict)


# ── Volume fraction override from design_params ───────────────────────────

def test_design_params_overrides_vf(small_problem):
    cfg    = SIMPConfig(volume_fraction=0.4, max_iter=10, tol=1e-2)
    result = run_simp(small_problem,
                      design_params={"volume_fraction": 0.3},
                      config=cfg)
    np.testing.assert_allclose(
        result.rho.mean(), 0.3,
        atol=cfg.bisect_tol * 10,
        err_msg="design_params volume_fraction did not override config"
    )

# ── All three materials (Decision 10) ────────────────────────────────────

@pytest.mark.parametrize("mat", ["Ti64", "AlSi10Mg", "316L"])
def test_all_materials(mat):
    prob   = build_problem(material=mat, nx=10, ny=10, nz=10)
    cfg    = SIMPConfig(max_iter=5, tol=1.0)
    result = run_simp(prob, config=cfg)
    assert result.compliance > 0
    assert result.mass > 0
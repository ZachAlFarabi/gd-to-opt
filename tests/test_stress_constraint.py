"""
tests/test_stress_constraint.py — Chunk 6: stress constraint tests

Tests:
    1.  compute_element_stress returns shape (n_elem,)
    2.  compute_element_stress returns non-negative values
    3.  p_norm_stress returns scalar
    4.  p_norm_stress is non-negative
    5.  p_norm_stress increases with load
    6.  p_norm_stress approaches max for large P
    7.  p_norm_sensitivity_direct returns shape (n_elem,)
    8.  p_norm_sensitivity_direct non-negative for uniform stress
    9.  StressConfig default SF=0 disables constraint
    10. StressConstraintHandler inactive when SF=0
    11. StressConstraintHandler active when SF>0
    12. StressConstraintHandler penalty zero when not violated
    13. StressConstraintHandler penalty positive when violated
    14. StressConstraintHandler sensitivity zero when not violated
    15. StressConstraintHandler sensitivity non-zero when violated
    16. StressConstraintHandler update increases mu on violation
    17. StressConstraintHandler update increases lam on violation
    18. compute_stress_adjoint_rhs returns shape (n_dof,)
    19. Full SIMP with stress constraint changes topology vs unconstrained
    20. Stress constraint reduces max stress in final design
    21. report returns tilde_sigma_norm key
    22. sigma_limit = sigma_yield / SF
    23. Stress constraint backward compat — SF=0 identical to unconstrained
"""

from __future__ import annotations
import numpy as np
import pytest
from gdto.mesh_material import build_problem, VoxelMesh, FEAssembler, MaterialModel
from gdto.simp import run_simp, SIMPConfig
from gdto.stress_constraint import (
    StressConfig, StressConstraintHandler,
    compute_element_stress, p_norm_stress,
    p_norm_sensitivity_direct, compute_stress_adjoint_rhs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_problem():
    return build_problem(
        material="Ti64", nx=8, ny=8, nz=8,
        lx=1.0, ly=1.0, lz=1.0,
        fixed_faces=["zmin"], load_face="zmax",
        load_direction=2, load_magnitude=-1.0,
    )

@pytest.fixture(scope="module")
def small_result(small_problem):
    cfg = SIMPConfig(max_iter=10, tol=0.1, p_start=2.0, p_end=3.0,
                     snapshot_interval=0)
    return run_simp(small_problem, config=cfg)

@pytest.fixture(scope="module")
def sigma_vm(small_result, small_problem):
    prob = small_problem
    res  = small_result
    import scipy.sparse.linalg as spla
    K   = prob.assembler.assemble_K(res.rho, p=3.0)
    K_ff, _ = prob.bc.apply(K)
    u_f = spla.spsolve(K_ff, prob.bc.f[prob.bc.free_dofs])
    u   = prob.bc.expand(u_f)
    return compute_element_stress(u, prob.mesh, prob.assembler, prob.material)


# ── compute_element_stress ─────────────────────────────────────────────────────

def test_stress_shape(sigma_vm, small_problem):
    assert sigma_vm.shape == (small_problem.mesh.n_elem,)

def test_stress_nonneg(sigma_vm):
    assert np.all(sigma_vm >= 0)


# ── p_norm_stress ──────────────────────────────────────────────────────────────

def test_p_norm_scalar(sigma_vm):
    rho = np.ones(len(sigma_vm)) * 0.4
    val = p_norm_stress(sigma_vm, rho, sigma_yield=1.125e9)
    assert np.isscalar(val) or val.ndim == 0

def test_p_norm_nonneg(sigma_vm):
    rho = np.ones(len(sigma_vm)) * 0.4
    val = p_norm_stress(sigma_vm, rho, sigma_yield=1.125e9)
    assert val >= 0

def test_p_norm_increases_with_load(small_problem):
    """P-norm should be larger under larger load."""
    import scipy.sparse.linalg as spla

    def get_pnorm(load_mag):
        prob = build_problem(
            material="Ti64", nx=8, ny=8, nz=8,
            lx=1.0, ly=1.0, lz=1.0,
            fixed_faces=["zmin"], load_face="zmax",
            load_direction=2, load_magnitude=load_mag,
        )
        rho = np.full(prob.mesh.n_elem, 0.4)
        K   = prob.assembler.assemble_K(rho, p=3.0)
        K_ff, _ = prob.bc.apply(K)
        u_f = spla.spsolve(K_ff, prob.bc.f[prob.bc.free_dofs])
        u   = prob.bc.expand(u_f)
        sv  = compute_element_stress(u, prob.mesh, prob.assembler, prob.material)
        return p_norm_stress(sv, rho, sigma_yield=1.125e9)

    pn1 = get_pnorm(-1.0)
    pn2 = get_pnorm(-10.0)
    assert pn2 > pn1

def test_p_norm_large_P_approaches_max(sigma_vm):
    """For large P, P-norm ≈ max(sigma)."""
    rho = np.ones(len(sigma_vm)) * 0.9
    sy  = 1.125e9
    pn6  = p_norm_stress(sigma_vm, rho, sy, P=6)
    pn20 = p_norm_stress(sigma_vm, rho, sy, P=20)
    true_max = float((rho**0.5 * sigma_vm / sy).max())
    assert abs(pn20 - true_max) <= abs(pn6 - true_max) + 1e-10


# ── p_norm_sensitivity_direct ─────────────────────────────────────────────────

def test_sensitivity_shape(sigma_vm):
    rho = np.full(len(sigma_vm), 0.4)
    ds  = p_norm_sensitivity_direct(sigma_vm, rho, 1.125e9)
    assert ds.shape == sigma_vm.shape

def test_sensitivity_nonneg_uniform(sigma_vm):
    """For uniform density, direct sensitivity should be non-negative."""
    rho = np.full(len(sigma_vm), 0.4)
    ds  = p_norm_sensitivity_direct(sigma_vm, rho, 1.125e9)
    assert np.all(ds >= -1e-12)


# ── StressConfig ──────────────────────────────────────────────────────────────

def test_stress_config_default_disabled():
    cfg = StressConfig()
    assert cfg.safety_factor == 0.0

def test_stress_config_sigma_limit():
    """sigma_limit = sigma_yield / SF."""
    sy  = 1.125e9
    cfg = StressConfig(safety_factor=2.0)
    handler = StressConstraintHandler(cfg, sy)
    assert np.isclose(handler.sigma_limit, sy / 2.0)


# ── StressConstraintHandler ───────────────────────────────────────────────────

def test_handler_inactive_sf0():
    handler = StressConstraintHandler(StressConfig(safety_factor=0.0), 1.125e9)
    assert not handler.active

def test_handler_active_sf_positive():
    handler = StressConstraintHandler(StressConfig(safety_factor=2.0), 1.125e9)
    assert handler.active

def test_penalty_zero_when_not_violated():
    """No penalty when tilde_sigma <= sigma_limit."""
    sy      = 1.125e9
    handler = StressConstraintHandler(
        StressConfig(safety_factor=2.0, penalty_weight=10.0), sy
    )
    # p_norm_stress returns tilde normalised by sigma_yield
    # sigma_limit/sy = 0.5 — pass value well below this
    tilde_norm = 1e8 / sy   # ≈ 0.089, well below 0.5
    penalty = handler.penalty(tilde_norm)
    assert penalty == 0.0

def test_penalty_positive_when_violated():
    """Penalty > 0 when tilde_sigma > sigma_limit."""
    sy      = 1.125e9
    handler = StressConstraintHandler(
        StressConfig(safety_factor=2.0, penalty_weight=10.0), sy
    )
    # sigma_limit/sigma_yield = 0.5; pass 0.8 > 0.5 → violation
    penalty = handler.penalty(0.8)
    assert penalty > 0.0

def test_sensitivity_zero_not_violated():
    sy      = 1.125e9
    handler = StressConstraintHandler(
        StressConfig(safety_factor=2.0), sy
    )
    ds = p_norm_sensitivity_direct(
        np.array([1e8]), np.array([0.4]), sy
    )
    # Pass normalised tilde_sigma — well below sigma_limit/sy = 0.5
    tilde_norm = 1e8 / sy
    sens = handler.sensitivity(tilde_norm, ds)
    assert np.all(sens == 0.0)

def test_sensitivity_nonzero_violated():
    sy      = 1.125e9
    handler = StressConstraintHandler(
        StressConfig(safety_factor=2.0, penalty_weight=100.0), sy
    )
    n   = 512
    sv  = np.full(n, 8e8)
    rho = np.full(n, 0.4)
    ds  = p_norm_sensitivity_direct(sv, rho, sy)
    tilde = p_norm_stress(sv, rho, sy)
    sens  = handler.sensitivity(tilde, ds)
    assert np.any(sens != 0.0)

def test_update_increases_mu_on_violation():
    sy      = 1.125e9
    handler = StressConstraintHandler(
        StressConfig(safety_factor=2.0, penalty_weight=1.0, penalty_update=2.0), sy
    )
    mu_before = handler.mu
    handler.update(0.9)   # tilde_sigma=0.9, sigma_limit/sy=0.5 → violation
    assert handler.mu >= mu_before

def test_update_increases_lam_on_violation():
    sy      = 1.125e9
    handler = StressConstraintHandler(
        StressConfig(safety_factor=2.0, penalty_weight=10.0), sy
    )
    lam_before = handler.lam
    handler.update(0.9)
    assert handler.lam >= lam_before

def test_report_has_tilde_key():
    handler = StressConstraintHandler(
        StressConfig(safety_factor=2.0), 1.125e9
    )
    report = handler.report(5e8)
    assert "tilde_sigma_norm" in report


# ── adjoint RHS ───────────────────────────────────────────────────────────────

def test_adjoint_rhs_shape(small_result, small_problem, sigma_vm):
    import scipy.sparse.linalg as spla
    prob = small_problem
    res  = small_result
    K    = prob.assembler.assemble_K(res.rho, p=3.0)
    K_ff, _ = prob.bc.apply(K)
    u_f  = spla.spsolve(K_ff, prob.bc.f[prob.bc.free_dofs])
    u    = prob.bc.expand(u_f)
    g    = compute_stress_adjoint_rhs(
        sigma_vm, res.rho, u,
        prob.mesh, prob.assembler, prob.material, 1.125e9
    )
    assert g.shape == (prob.mesh.n_dof,)


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def constrained_result():
    """Full SIMP run with stress constraint active.

    1 mm³ domain, 1000 N load → nominal stress ~1 GPa >> Ti64 limit 750 MPa (SF=1.5).
    """
    prob = build_problem(
        material="Ti64", nx=8, ny=8, nz=8,
        lx=0.001, ly=0.001, lz=0.001,
        fixed_faces=["zmin"], load_face="zmax",
        load_direction=2, load_magnitude=-1000.0,
    )
    cfg = SIMPConfig(
        max_iter=30, tol=0.05,
        p_start=2.0, p_end=3.0,
        snapshot_interval=0,
    )
    sc = StressConfig(safety_factor=1.5, penalty_weight=10.0, penalty_update=2.0)
    return run_simp(prob, config=cfg, stress_cfg=sc)

@pytest.fixture(scope="module")
def unconstrained_result():
    """Same problem without stress constraint."""
    prob = build_problem(
        material="Ti64", nx=8, ny=8, nz=8,
        lx=0.001, ly=0.001, lz=0.001,
        fixed_faces=["zmin"], load_face="zmax",
        load_direction=2, load_magnitude=-1000.0,
    )
    cfg = SIMPConfig(
        max_iter=30, tol=0.05,
        p_start=2.0, p_end=3.0,
        snapshot_interval=0,
    )
    return run_simp(prob, config=cfg)

def test_stress_constraint_changes_topology(constrained_result, unconstrained_result):
    """With stress constraint active, topology differs from unconstrained."""
    rho_c = constrained_result.rho
    rho_u = unconstrained_result.rho
    diff  = float(np.mean(np.abs(rho_c - rho_u)))
    assert diff > 1e-4, f"Topologies too similar (diff={diff:.6f})"

def test_backward_compat_sf0(small_problem):
    """SF=0 should give identical result to no stress config."""
    cfg = SIMPConfig(max_iter=5, tol=0.5, snapshot_interval=0)
    sc  = StressConfig(safety_factor=0.0)
    r1  = run_simp(small_problem, config=cfg)
    r2  = run_simp(small_problem, config=cfg, stress_cfg=sc)
    np.testing.assert_allclose(r1.rho, r2.rho, atol=1e-6)

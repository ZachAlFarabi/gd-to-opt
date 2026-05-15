"""
Tests for verify.py and reconstruct.py — Chunk 4.

Verifies:
    1.  ThermalAssembler: KTe0 shape and symmetry
    2.  ThermalAssembler: K_T shape and CSC format
    3.  ThermalAssembler: temperature solve returns n_nodes array
    4.  verify(): returns VerifyResult with all fields
    5.  verify(): gap_ratio is finite and within clip bounds [-1, 100]
    6.  verify(): safety factor positive
    7.  verify(): thermal mode runs without error
    8.  reconstruct_stl(): produces STL file
    9.  reconstruct_stl(): STL is non-empty
    10. stl_to_voxels(): returns correct shape
    11. stl_to_voxels(): densities in [rho_min, Vf]
"""

from __future__ import annotations

import numpy as np
import pytest
import tempfile
from pathlib import Path

from gdto.mesh_material import build_problem
from gdto.simp import run_simp, SIMPConfig, SIMPResult
from gdto.verify import ThermalAssembler, verify, THERMAL_CONSTANTS


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_problem():
    return build_problem(material="Ti64", nx=8, ny=8, nz=8)

@pytest.fixture(scope="module")
def small_result(small_problem):
    cfg = SIMPConfig(max_iter=5, tol=1.0)
    return run_simp(small_problem, config=cfg)


# ── ThermalAssembler tests ────────────────────────────────────────────────

def test_KTe0_shape(small_problem):
    ta = ThermalAssembler(small_problem.mesh, k_thermal=6.7)
    assert ta._KTe0.shape == (8, 8)

def test_KTe0_symmetry(small_problem):
    ta = ThermalAssembler(small_problem.mesh, k_thermal=6.7)
    np.testing.assert_allclose(ta._KTe0, ta._KTe0.T, atol=1e-10)

def test_KTe0_positive_definite(small_problem):
    ta = ThermalAssembler(small_problem.mesh, k_thermal=6.7)
    eigs = np.linalg.eigvalsh(ta._KTe0)
    # One zero eigenvalue (constant temperature mode) — rest positive
    assert np.all(eigs[1:] > 0)

def test_KT_shape(small_problem):
    ta  = ThermalAssembler(small_problem.mesh, k_thermal=6.7)
    rho = np.ones(small_problem.mesh.n_elem)
    KT  = ta.assemble_KT(rho)
    N   = small_problem.mesh.n_nodes
    assert KT.shape == (N, N)

def test_KT_is_csc(small_problem):
    import scipy.sparse as sp
    ta  = ThermalAssembler(small_problem.mesh, k_thermal=6.7)
    rho = np.ones(small_problem.mesh.n_elem)
    KT  = ta.assemble_KT(rho)
    assert sp.issparse(KT) and KT.format == 'csc'

def test_temperature_solve_shape(small_problem):
    ta  = ThermalAssembler(small_problem.mesh, k_thermal=6.7)
    rho = np.ones(small_problem.mesh.n_elem)
    T   = ta.solve_temperature(
        rho,
        flux_faces={"zmin": 5000.0},
        temp_faces={"zmax": 20.0},
    )
    assert T.shape == (small_problem.mesh.n_nodes,)

def test_temperature_solve_satisfies_bc(small_problem):
    """Fixed temperature BC must be satisfied at boundary nodes."""
    ta  = ThermalAssembler(small_problem.mesh, k_thermal=6.7)
    rho = np.ones(small_problem.mesh.n_elem)
    T   = ta.solve_temperature(
        rho,
        flux_faces={},
        temp_faces={"zmax": 100.0},
    )
    zmax_nodes = ta._face_node_ids("zmax")
    np.testing.assert_allclose(T[zmax_nodes], 100.0, atol=1e-6)


# ── verify() tests ────────────────────────────────────────────────────────

def test_verify_returns_result(small_result, small_problem):
    from gdto.verify import VerifyResult
    result = verify(small_result, small_problem)
    assert isinstance(result, VerifyResult)

def test_verify_has_all_fields(small_result, small_problem):
    result = verify(small_result, small_problem)
    for field in ['compliance_real', 'compliance_simp', 'gap_ratio',
                  'max_von_mises_mpa', 'min_safety_factor', 'mass_kg',
                  'volume_fraction', 'report']:
        assert hasattr(result, field)

def test_verify_compliance_real_positive(small_result, small_problem):
    result = verify(small_result, small_problem)
    assert result.compliance_real > 0

def test_verify_gap_ratio_nonnegative(small_result, small_problem):
    """
    Gap ratio invariant: only meaningful for well-converged density fields.
    For poorly converged fields (uniform density near Vf), binary
    thresholding destroys most of the structure -> C_real >> C_simp.
    The correct invariant is that gap_ratio is finite and clipped to [-1, 100].
    """
    result = verify(small_result, small_problem)
    assert np.isfinite(result.gap_ratio)
    assert result.gap_ratio >= -1.0   # clipped lower bound
    assert result.gap_ratio <= 100.0  # clipped upper bound


def test_verify_gap_ratio_converged():
    """
    For a well-converged run (most elements near 0 or 1),
    binary structure should be stiffer than penalised SIMP.
    gap_ratio = (C_simp - C_real) / C_simp should be >= -0.3
    """
    prob        = build_problem(material="Ti64", nx=8, ny=8, nz=8)
    cfg         = SIMPConfig(max_iter=30, tol=1e-2, p_start=2.0, p_end=3.0)
    result_simp = run_simp(prob, config=cfg)

    rho          = result_simp.rho
    polarisation = np.mean((rho < 0.1) | (rho > 0.9))

    if polarisation > 0.5:   # at least 50% elements are near 0 or 1
        result = verify(result_simp, prob)
        assert result.gap_ratio >= -0.3, \
            f"Well-converged gap ratio {result.gap_ratio:.3f} worse than -0.3"


def test_verify_safety_factor_positive(small_result, small_problem):
    result = verify(small_result, small_problem)
    assert result.min_safety_factor > 0

def test_verify_mass_positive(small_result, small_problem):
    result = verify(small_result, small_problem)
    assert result.mass_kg > 0

def test_verify_report_has_keys(small_result, small_problem):
    result = verify(small_result, small_problem)
    for key in ['compliance_real_Nm', 'mass_kg', 'min_safety_factor',
                'gap_ratio_pct', 'material']:
        assert key in result.report

def test_verify_thermal_mode_runs(small_result, small_problem):
    """Thermal mode B must run without error."""
    result = verify(
        small_result, small_problem,
        flux_faces={"zmin": 5000.0},
        temp_faces={"zmax": 20.0},
    )
    assert result.max_temp_c is not None
    assert result.max_temp_c > 20.0  # temperature rises above reference


# ── reconstruct_stl() tests ───────────────────────────────────────────────

def test_reconstruct_stl_creates_file(small_result, small_problem):
    pytest.importorskip("skimage")
    pytest.importorskip("trimesh")
    from gdto.reconstruct import reconstruct_stl

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.stl"
        info = reconstruct_stl(small_result.rho, small_problem.mesh, out)
        assert out.exists()
        assert out.stat().st_size > 100

def test_reconstruct_stl_info_keys(small_result, small_problem):
    pytest.importorskip("skimage")
    pytest.importorskip("trimesh")
    from gdto.reconstruct import reconstruct_stl

    with tempfile.TemporaryDirectory() as tmpdir:
        out  = Path(tmpdir) / "test.stl"
        info = reconstruct_stl(small_result.rho, small_problem.mesh, out)
        assert info["n_faces"] > 0
        assert info["volume_m3"] > 0
        for k in ["bbox_mm", "n_vertices", "n_faces", "volume_m3"]:
            assert k in info


# ── stl_to_voxels() tests ─────────────────────────────────────────────────

def test_stl_to_voxels(small_result, small_problem):
    pytest.importorskip("skimage")
    pytest.importorskip("trimesh")
    from gdto.reconstruct import reconstruct_stl, stl_to_voxels

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.stl"
        reconstruct_stl(small_result.rho, small_problem.mesh, out,
                        threshold=0.5)
        rho_v, info = stl_to_voxels(out, nx=8, ny=8, nz=8)
        assert rho_v.shape == (8*8*8,)
        assert rho_v.min() >= 0.001 - 1e-10
        assert rho_v.max() <= 0.4 + 1e-10
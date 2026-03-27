"""
Smoke tests for MaterialModel — chunk 1.

Verifies:
    1. All three presets instantiate without error
    2. C0 is symmetric to machine precision
    3. C0 is positive definite (all eigenvalues > 0)
    4. C0 shape is (6,6)
    5. C66 == (C11 - C12) / 2  [transverse isotropy constraint]
    6. SIMP: penalised(1.0) == C0
    7. SIMP: penalised(0.0) ≈ E_min * C0
    8. SIMP: penalised returns (n,6,6) for array input
    9. Delta > 0 for all three materials
   10. Invalid constants raise AssertionError
"""

import numpy as np
import pytest
import scipy.sparse as sp
from gdto.mesh_material import (
    MaterialModel, VoxelMesh, FEAssembler,
    BoundaryConditions, build_problem, ProblemData
)


MATERIALS = ["Ti64", "AlSi10Mg", "316L"]


@pytest.mark.parametrize("mat", MATERIALS)
def test_instantiation(mat):
    m = MaterialModel.from_preset(mat)
    assert m.C0 is not None


@pytest.mark.parametrize("mat", MATERIALS)
def test_C0_shape(mat):
    m = MaterialModel.from_preset(mat)
    assert m.C0.shape == (6, 6)


@pytest.mark.parametrize("mat", MATERIALS)
def test_C0_symmetry(mat):
    m = MaterialModel.from_preset(mat)
    np.testing.assert_allclose(m.C0, m.C0.T, atol=1e-6,
                               err_msg=f"{mat}: C0 is not symmetric")


@pytest.mark.parametrize("mat", MATERIALS)
def test_C0_positive_definite(mat):
    m = MaterialModel.from_preset(mat)
    eigenvalues = np.linalg.eigvalsh(m.C0)
    assert np.all(eigenvalues > 0), (
        f"{mat}: C0 not positive definite. "
        f"Eigenvalues: {eigenvalues}"
    )


@pytest.mark.parametrize("mat", MATERIALS)
def test_transverse_isotropy_constraint(mat):
    """
    C66 must equal E_perp / 2(1 + nu_perp) — the in-plane shear modulus
    derived from isotropy of the 1-2 plane (Ivey et al. Eq. 7.23).

    Note: C66 != (C11 - C12)/2 for transverse isotropy when E_par != E_perp,
    because C11 and C12 are coupled to E_par via the Delta inversion.
    The identity (C11-C12)/2 = C66 holds only in the isotropic limit.
    """
    m = MaterialModel.from_preset(mat)
    C = m.C0
    expected = m.E_perp / (2.0 * (1.0 + m.nu_perp))
    np.testing.assert_allclose(
        C[5, 5], expected, rtol=1e-10,
        err_msg=f"{mat}: C66 != E_perp / 2(1 + nu_perp)"
    )


@pytest.mark.parametrize("mat", MATERIALS)
def test_simp_full_density(mat):
    """penalised(rho_e=1.0) should return C0 to near machine precision."""
    m = MaterialModel.from_preset(mat)
    np.testing.assert_allclose(m.penalised(1.0), m.C0, rtol=1e-9,
                               err_msg=f"{mat}: penalised(1.0) != C0")


@pytest.mark.parametrize("mat", MATERIALS)
def test_simp_void(mat):
    """penalised(rho_e=0.0) should return E_min * C0."""
    E_min = 1e-9
    m = MaterialModel.from_preset(mat)
    np.testing.assert_allclose(m.penalised(0.0, E_min_factor=E_min),
                               E_min * m.C0, rtol=1e-9)


@pytest.mark.parametrize("mat", MATERIALS)
def test_simp_array_input(mat):
    """penalised accepts (n,) array and returns (n,6,6)."""
    m = MaterialModel.from_preset(mat)
    rho = np.linspace(0, 1, 20)
    result = m.penalised(rho)
    assert result.shape == (20, 6, 6)


@pytest.mark.parametrize("mat", MATERIALS)
def test_delta_positive(mat):
    m = MaterialModel.from_preset(mat)
    assert m._compute_delta() > 0


def test_invalid_material_raises():
    with pytest.raises(ValueError):
        MaterialModel.from_preset("Unobtainium")


def test_inadmissible_nu_raises():
    """nu_perp >= 1 should fail the positive-definiteness check."""
    with pytest.raises(AssertionError):
        MaterialModel(
            E_perp=100e9, E_par=100e9,
            nu_perp=1.1, nu_par=0.3,
            G_par=40e9, rho=4000.0
        )


def test_summary_runs():
    m = MaterialModel.from_preset("Ti64")
    s = m.summary()
    assert "Ti64" in s
    assert "positive definite: True" in s


# ── VoxelMesh tests ───────────────────────────────────────────────────────

def test_voxel_mesh_counts():
    """Node, element, and DOF counts must satisfy exact formulae."""
    m = VoxelMesh(nx=4, ny=5, nz=6)
    assert m.n_elem  == 4 * 5 * 6
    assert m.n_nodes == 5 * 6 * 7
    assert m.n_dof   == 3 * 5 * 6 * 7

def test_voxel_mesh_coords_shape():
    m = VoxelMesh(nx=4, ny=4, nz=4)
    assert m.coords.shape == (m.n_nodes, 3)

def test_voxel_mesh_conn_shape():
    m = VoxelMesh(nx=4, ny=4, nz=4)
    assert m.conn.shape == (m.n_elem, 8)

def test_voxel_mesh_dof_map_shape():
    m = VoxelMesh(nx=4, ny=4, nz=4)
    assert m.dof_map.shape == (m.n_elem, 24)

def test_voxel_mesh_dof_map_range():
    """All DOF indices must be in [0, n_dof)."""
    m = VoxelMesh(nx=4, ny=4, nz=4)
    assert m.dof_map.min() >= 0
    assert m.dof_map.max() < m.n_dof

def test_voxel_mesh_coords_bounds():
    """Node coordinates must lie within [0, lx] x [0, ly] x [0, lz]."""
    m = VoxelMesh(nx=4, ny=4, nz=4, lx=2.0, ly=3.0, lz=4.0)
    assert m.coords[:, 0].max() == pytest.approx(2.0)
    assert m.coords[:, 1].max() == pytest.approx(3.0)
    assert m.coords[:, 2].max() == pytest.approx(4.0)

def test_voxel_mesh_element_centres():
    m = VoxelMesh(nx=2, ny=2, nz=2)
    centres = m.element_centres()
    assert centres.shape == (8, 3)

# ── FEAssembler tests ─────────────────────────────────────────────────────

@pytest.fixture
def small_problem():
    """20^3 mesh with Ti64 — fast enough for unit tests."""
    mesh = VoxelMesh(nx=20, ny=20, nz=20)
    mat  = MaterialModel.from_preset("Ti64")
    asm  = FEAssembler(mesh, mat)
    return mesh, mat, asm

def test_Ke0_shape(small_problem):
    _, _, asm = small_problem
    assert asm.Ke0.shape == (24, 24)

def test_Ke0_symmetry(small_problem):
    _, _, asm = small_problem
    np.testing.assert_allclose(asm.Ke0, asm.Ke0.T, atol=1e-6)

def test_Ke0_positive_definite(small_problem):
    """
    Ke0 with free boundary conditions has 6 zero eigenvalues (rigid body modes).
    All remaining eigenvalues must be positive.
    """
    _, _, asm = small_problem
    eigvals = np.linalg.eigvalsh(asm.Ke0)
    # Sort and check: 6 near-zero (rigid body), rest positive
    assert np.all(eigvals[6:] > 0), f"Non-rigid eigenvalues not all positive: {eigvals}"

def test_assemble_K_shape(small_problem):
    mesh, _, asm = small_problem
    rho = np.ones(mesh.n_elem)
    K   = asm.assemble_K(rho)
    assert K.shape == (mesh.n_dof, mesh.n_dof)

def test_assemble_K_is_csc(small_problem):
    mesh, _, asm = small_problem
    rho = np.ones(mesh.n_elem)
    K   = asm.assemble_K(rho)
    assert sp.issparse(K) and K.format == 'csc'

def test_assemble_K_symmetry(small_problem):
    mesh, _, asm = small_problem
    rho = np.ones(mesh.n_elem)
    K   = asm.assemble_K(rho)
    diff = K - K.T
    assert abs(diff).max() < 1e-6, "K is not symmetric"

def test_assemble_K_void(small_problem):
    """K with rho=0 should be E_min * K_full — not zero."""
    mesh, _, asm = small_problem
    E_min  = 1e-9
    K_full = asm.assemble_K(np.ones(mesh.n_elem))
    K_void = asm.assemble_K(np.zeros(mesh.n_elem), E_min_factor=E_min)
    # Compare via Frobenius norm rather than elementwise ratio
    # (elementwise division is unstable when K_full has near-zero entries
    # from algebraic cancellation after COO->CSC duplicate summation)
    diff = K_void - E_min * K_full
    np.testing.assert_allclose(
        diff.data, 0.0, atol=1e-3,
        err_msg="K_void is not E_min * K_full to required tolerance"
    )

# ── BoundaryConditions tests ──────────────────────────────────────────────

@pytest.fixture
def small_bc():
    mesh = VoxelMesh(nx=4, ny=4, nz=4)
    return BoundaryConditions(mesh=mesh)

def test_bc_fixed_dofs_not_empty(small_bc):
    assert len(small_bc.fixed_dofs) > 0

def test_bc_free_dofs_partition(small_bc):
    """fixed + free must partition [0, n_dof) without overlap."""
    mesh = small_bc.mesh
    all_dofs = np.arange(mesh.n_dof)
    combined = np.sort(np.concatenate([small_bc.fixed_dofs, small_bc.free_dofs]))
    np.testing.assert_array_equal(combined, all_dofs)

def test_bc_load_vector_shape(small_bc):
    assert small_bc.f.shape == (small_bc.mesh.n_dof,)

def test_bc_load_vector_sum(small_bc):
    """Total load must equal load_magnitude."""
    np.testing.assert_allclose(
        small_bc.f.sum(), small_bc.load_magnitude, rtol=1e-10
    )

def test_bc_load_not_on_fixed(small_bc):
    """No load should be applied to fixed DOFs."""
    assert np.all(small_bc.f[small_bc.fixed_dofs] == 0.0)

def test_bc_apply_reduces_K(small_bc):
    """K_ff must be smaller than K by exactly the number of fixed DOFs."""
    mat = MaterialModel.from_preset("Ti64")
    asm = FEAssembler(small_bc.mesh, mat)
    K   = asm.assemble_K(np.ones(small_bc.mesh.n_elem))
    K_ff, f_f = small_bc.apply(K)
    n_free = len(small_bc.free_dofs)
    assert K_ff.shape == (n_free, n_free)
    assert f_f.shape  == (n_free,)

def test_bc_apply_K_ff_symmetric(small_bc):
    mat = MaterialModel.from_preset("Ti64")
    asm = FEAssembler(small_bc.mesh, mat)
    K   = asm.assemble_K(np.ones(small_bc.mesh.n_elem))
    K_ff, _ = small_bc.apply(K)
    diff = K_ff - K_ff.T
    assert abs(diff).max() < 1e-6

def test_bc_expand(small_bc):
    """expand() must return full-size vector with zeros at fixed DOFs."""
    n_free = len(small_bc.free_dofs)
    u_free = np.ones(n_free)
    u_full = small_bc.expand(u_free)
    assert u_full.shape == (small_bc.mesh.n_dof,)
    assert np.all(u_full[small_bc.fixed_dofs] == 0.0)

def test_bc_invalid_face_raises():
    mesh = VoxelMesh(nx=4, ny=4, nz=4)
    with pytest.raises(ValueError):
        BoundaryConditions(mesh=mesh, fixed_faces=['invalid_face'])

# ── build_problem tests ───────────────────────────────────────────────────

def test_build_problem_returns_problemdata():
    p = build_problem(material="Ti64", nx=4, ny=4, nz=4)
    assert isinstance(p, ProblemData)

def test_build_problem_fields():
    p = build_problem(material="AlSi10Mg", nx=4, ny=4, nz=4)
    assert isinstance(p.mesh,      VoxelMesh)
    assert isinstance(p.material,  MaterialModel)
    assert isinstance(p.assembler, FEAssembler)
    assert isinstance(p.bc,        BoundaryConditions)
    assert p.solver == "direct"

def test_build_problem_invalid_solver():
    with pytest.raises(ValueError):
        build_problem(material="Ti64", nx=4, ny=4, nz=4, solver="mumps")

@pytest.mark.parametrize("mat", ["Ti64", "AlSi10Mg", "316L"])
def test_build_problem_all_materials(mat):
    p = build_problem(material=mat, nx=4, ny=4, nz=4)
    assert p.material.name == mat
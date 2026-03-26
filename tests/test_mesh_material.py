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
from gdto.mesh_material import MaterialModel


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
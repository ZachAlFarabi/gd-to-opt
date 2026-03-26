"""
mesh_material.py — Chunk 1: material model

Constitutive tensor C_ijkl in Voigt notation for transverse isotropic
LPBF metals. Symmetry axis = 3 (build direction Z).

Theoretical basis:
    Ivey, Carey & Ayranci (2017), Chapter 7 in Handbook of Advances in
    Braided Composite Materials, Elsevier/Woodhead Publishing.
    Equations (7.23), (7.50)-(7.60) specialised to transverse isotropy
    with isotropic plane = 1-2 (XY), unique axis = 3 (Z).

Source convention:
    E_perp  = E_XY  — Young's modulus in the isotropic XY plane
    E_par   = E_Z   — Young's modulus along the build axis Z
    nu_perp = nu_12 — in-plane Poisson ratio (Ivey et al. Table 7.2)
    nu_par  = nu_13 — cross-plane Poisson ratio
    G_par   = G_13  — out-of-plane shear modulus
    C66     = E_perp / 2(1 + nu_perp)  — derived, not independent
                                          (Ivey et al. Eq. 7.23 constraint)

Data sources:
    E_perp, E_par, rho — user CSV (as-built XY/Z rows)
    nu_perp, nu_par, G_par — LPBF literature (see configs/default.yaml)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Preset material databases
# ---------------------------------------------------------------------------

# Engineering constants for as-built LPBF materials.
# E values from user CSV (as-built); nu and G from LPBF literature.
# See Decision 2 derivation for full provenance.
_PRESETS: dict[str, dict] = {
    "Ti64": {
        # Source: user CSV rows 9-10 (as-built XY/Z Young's modulus)
        "E_perp":  108e9,   # Pa — XY plane, ±20 GPa
        "E_par":   105e9,   # Pa — Z axis,  ±20 GPa
        # Source: Pantawane et al. 2022, MDPI Materials
        # pulse-echo ultrasonic on SLM Ti-6Al-4V
        "nu_perp": 0.342,
        "nu_par":  0.330,
        "G_par":   47e9,    # Pa
        # Source: user CSV row 3 right block
        "rho":     4400.0,  # kg/m³
    },
    "AlSi10Mg": {
        # Source: user CSV rows 9-10 (as-built XY/Z Young's modulus)
        "E_perp":  75e9,    # Pa — XY plane, ±10 GPa
        "E_par":   70e9,    # Pa — Z axis,  ±30 GPa  <-- large uncertainty
        # Source: Sert et al. 2019, lateral strain sensor tensile tests SLM
        "nu_perp": 0.330,
        "nu_par":  0.310,
        "G_par":   25e9,    # Pa
        # Source: user CSV row 3 right block
        "rho":     2680.0,  # kg/m³
    },
    "316L": {
        # Source: user CSV rows 9-10 (as-built XY/Z Young's modulus)
        "E_perp":  197e9,   # Pa — XY plane, ±5 GPa
        "E_par":   190e9,   # Pa — Z axis,  ±10 GPa
        # Source: Li et al. 2021 (PMC8595114), Chao et al. 2022
        # transverse isotropic constitutive fit for SLM 316L
        "nu_perp": 0.280,
        "nu_par":  0.260,
        "G_par":   77e9,    # Pa
        # Source: user CSV row 3 right block
        "rho":     7990.0,  # kg/m³
    },
}


# ---------------------------------------------------------------------------
# MaterialModel
# ---------------------------------------------------------------------------

@dataclass
class MaterialModel:
    """
    Transverse isotropic material model for LPBF metals.

    The 6x6 Voigt stiffness matrix C0 is computed from 5 independent
    engineering constants via closed-form inversion of the compliance
    matrix S (Ivey et al. Eqs. 7.51-7.60 specialised to TI symmetry).

    SIMP penalisation is applied via C(rho_e) = rho_e^p * C0, where
    C0 is the full-density stiffness computed here.

    Parameters
    ----------
    E_perp : float
        Young's modulus in the isotropic XY plane [Pa]. From user CSV.
    E_par : float
        Young's modulus along the build axis Z [Pa]. From user CSV.
    nu_perp : float
        In-plane Poisson ratio (dimensionless). From LPBF literature.
    nu_par : float
        Cross-plane Poisson ratio (dimensionless). From LPBF literature.
    G_par : float
        Out-of-plane shear modulus [Pa]. From LPBF literature.
    rho : float
        Mass density [kg/m³]. From user CSV.
    name : str
        Material identifier for logging.
    """

    E_perp:  float
    E_par:   float
    nu_perp: float
    nu_par:  float
    G_par:   float
    rho:     float
    name:    str = "custom"

    # Computed on post-init — not constructor arguments
    C0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_inputs()
        self.C0 = self._build_C_voigt()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, material: Literal["Ti64", "AlSi10Mg", "316L"]) -> "MaterialModel":
        """
        Instantiate from a built-in LPBF material preset.

        Parameters
        ----------
        material : str
            One of 'Ti64', 'AlSi10Mg', '316L'.
        """
        if material not in _PRESETS:
            raise ValueError(
                f"Unknown material '{material}'. "
                f"Available: {list(_PRESETS.keys())}"
            )
        return cls(**_PRESETS[material], name=material)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """
        Check thermodynamic admissibility (positive-definiteness of S).

        Requirements (Ivey et al. §7.3.1.4 specialised):
            E_perp, E_par, G_par > 0
            |nu_perp| < 1
            |nu_par| < sqrt(E_par / E_perp)
            Delta > 0  where Delta = 1 - nu_perp^2 - 2*nu_par^2*(E_par/E_perp)
        """
        assert self.E_perp > 0 and self.E_par > 0 and self.G_par > 0, \
            "Moduli must be positive."
        assert abs(self.nu_perp) < 1.0, \
            f"|nu_perp| = {abs(self.nu_perp):.3f} must be < 1."

        nu_par_bound = (self.E_par / self.E_perp) ** 0.5
        assert abs(self.nu_par) < nu_par_bound, (
            f"|nu_par| = {abs(self.nu_par):.3f} must be < "
            f"sqrt(E_par/E_perp) = {nu_par_bound:.3f}."
        )

        delta = self._compute_delta()
        assert delta > 0, (
            f"Positive-definiteness violated: Delta = {delta:.6f} <= 0. "
            f"Check nu_perp={self.nu_perp}, nu_par={self.nu_par}, "
            f"E_par/E_perp={self.E_par/self.E_perp:.4f}."
        )

    def _compute_delta(self) -> float:
        """
        Compute the determinant scalar Delta used in C inversion.

        Delta = 1 - nu_perp^2 - 2*nu_par^2*(E_par/E_perp)

        Must be > 0 for S to be positive definite.
        """
        return (
            1.0
            - self.nu_perp ** 2
            - 2.0 * self.nu_par ** 2 * (self.E_par / self.E_perp)
        )

    # ------------------------------------------------------------------
    # Core computation: C0
    # ------------------------------------------------------------------

    def _build_C_voigt(self) -> np.ndarray:
        """
        Compute the 6x6 Voigt stiffness matrix C0 from the 5 independent
        engineering constants.

        Derivation: closed-form inversion of the transverse isotropic
        compliance matrix S. See Ivey et al. (2017) Eqs. (7.51)-(7.60)
        specialised to TI symmetry with isotropic plane = 1-2:

            S11 = S22 = 1/E_perp
            S33 = 1/E_par
            S12 = -nu_perp/E_perp
            S13 = S23 = -nu_par/E_par
            S44 = S55 = 1/G_par
            S66 = 2(1+nu_perp)/E_perp  [derived from isotropy of 1-2 plane]

        Closed-form C components (Ivey et al. Eqs. 7.61-7.66 specialised):

            Delta = 1 - nu_perp^2 - 2*nu_par^2*(E_par/E_perp)

            C11 = E_perp * (1 - nu_par^2 * E_par/E_perp) / Delta
            C33 = E_par  * (1 - nu_perp^2)               / Delta
            C12 = E_perp * (nu_perp + nu_par^2*E_par/E_perp) / Delta
            C13 = E_par  * nu_par*(1 + nu_perp)           / Delta
            C44 = G_par                        [literature, independent]
            C66 = E_perp / 2(1+nu_perp)       [derived from 1-2 isotropy]

        Returns
        -------
        C0 : np.ndarray, shape (6, 6)
            Full-density Voigt stiffness matrix in Pa.
        """
        Ep  = self.E_perp
        Ez  = self.E_par
        nxy = self.nu_perp
        nxz = self.nu_par
        Gxz = self.G_par

        delta = self._compute_delta()  # dimensionless, pre-validated > 0

        # Normal stiffness components — all units Pa
        C11 = Ep * (1.0 - nxz**2 * (Ez/Ep)) / delta
        C33 = Ez * (1.0 - nxy**2)            / delta
        C12 = Ep * (nxy + nxz**2 * (Ez/Ep)) / delta
        C13 = Ez * (nxz * (1.0 + nxy))       / delta

        # Shear stiffness components
        C44 = Gxz                          # out-of-plane: from literature
        C66 = Ep / (2.0 * (1.0 + nxy))    # in-plane: derived from isotropy

        # Assemble symmetric 6x6
        C0 = np.zeros((6, 6), dtype=np.float64)
        C0[0, 0] = C0[1, 1] = C11
        C0[2, 2]             = C33
        C0[0, 1] = C0[1, 0] = C12
        C0[0, 2] = C0[2, 0] = C13
        C0[1, 2] = C0[2, 1] = C13
        C0[3, 3] = C0[4, 4] = C44
        C0[5, 5]             = C66

        return C0

    # ------------------------------------------------------------------
    # SIMP interface — called by the inner loop assembler
    # ------------------------------------------------------------------

    def penalised(self, rho_e: float | np.ndarray, p: float = 3.0,
                  E_min_factor: float = 1e-9) -> np.ndarray:
        """
        Return the SIMP-penalised stiffness tensor for element density rho_e.

        C(rho_e) = (E_min + rho_e^p * (1 - E_min)) * C0

        where E_min = E_min_factor * E_perp prevents singularity when
        rho_e -> 0. This is the modified SIMP scheme (Sigmund 2007).

        Parameters
        ----------
        rho_e : float or np.ndarray
            Element density, in [0, 1]. Scalar or array of shape (n_elem,).
        p : float
            Penalisation exponent. Default 3 (standard SIMP).
        E_min_factor : float
            Void stiffness floor as fraction of E_perp. Default 1e-9.

        Returns
        -------
        np.ndarray
            Shape (6, 6) if rho_e is scalar, or (n_elem, 6, 6) if array.
        """
        E_min = E_min_factor  # normalised — C0 already in Pa
        scale = E_min + np.asarray(rho_e) ** p * (1.0 - E_min)

        if np.ndim(scale) == 0:
            return float(scale) * self.C0
        else:
            # Broadcast: (n_elem,) * (6,6) -> (n_elem, 6, 6)
            return scale[:, np.newaxis, np.newaxis] * self.C0[np.newaxis]

    # ------------------------------------------------------------------
    # Build-orientation rotation hook (implemented in chunk 7)
    # ------------------------------------------------------------------

    def rotated(self, theta: float, axis: str = "y") -> "MaterialModel":
        """
        Return a new MaterialModel with C0 rotated by angle theta about axis.

        Stub — full Bond transformation matrix implemented in chunk 7
        (anisotropy.py). Calling this before chunk 7 raises NotImplementedError.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        axis : str
            Rotation axis, one of 'x', 'y', 'z'.
        """
        raise NotImplementedError(
            "Build-orientation rotation is implemented in chunk 7 "
            "(src/gdto/anisotropy.py). This stub will be replaced there."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the material constants and C0."""
        delta = self._compute_delta()
        eigenvalues = np.linalg.eigvalsh(self.C0)
        lines = [
            f"MaterialModel: {self.name}",
            f"  E_perp  = {self.E_perp/1e9:.1f} GPa  (from CSV)",
            f"  E_par   = {self.E_par/1e9:.1f} GPa  (from CSV)",
            f"  nu_perp = {self.nu_perp:.3f}        (literature)",
            f"  nu_par  = {self.nu_par:.3f}        (literature)",
            f"  G_par   = {self.G_par/1e9:.1f} GPa  (literature)",
            f"  rho     = {self.rho:.0f} kg/m³ (from CSV)",
            f"  Delta   = {delta:.4f}  (must be > 0)",
            f"  C0 eigenvalues (GPa): "
            f"{', '.join(f'{v/1e9:.1f}' for v in eigenvalues)}",
            f"  C0 positive definite: {np.all(eigenvalues > 0)}",
        ]
        return "\n".join(lines)
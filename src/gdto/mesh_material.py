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

# ── Add these imports at the top of mesh_material.py ─────────────────────
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import scipy.sparse as sp



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
    
# ---------------------------------------------------------------------------
# VoxelMesh
# ---------------------------------------------------------------------------

@dataclass
class VoxelMesh:
    """
    Regular structured hexahedral (brick) mesh over a unit domain [0,1]^3.

    The mesh is parameterised by (nx, ny, nz) — the number of elements
    along each axis. Node and element numbering follow C-order (z fastest).

    Coordinate convention
    ---------------------
    Axis 1 = X (width),  n_nodes_x = nx + 1
    Axis 2 = Y (depth),  n_nodes_y = ny + 1
    Axis 3 = Z (height / build direction), n_nodes_z = nz + 1

    The build direction is always axis 3 (Z), consistent with the
    MaterialModel convention (unique axis = 3).

    Parameters
    ----------
    nx, ny, nz : int
        Number of elements along X, Y, Z. Default 40 each.
    lx, ly, lz : float
        Physical domain size [m]. Default 1.0 each (unit cube).
    """

    nx: int = 40
    ny: int = 40
    nz: int = 40
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0

    # Computed in post-init
    n_elem:  int        = field(init=False)
    n_nodes: int        = field(init=False)
    n_dof:   int        = field(init=False)
    coords:  np.ndarray = field(init=False, repr=False)  # (n_nodes, 3)
    conn:    np.ndarray = field(init=False, repr=False)  # (n_elem, 8)
    dof_map: np.ndarray = field(init=False, repr=False)  # (n_elem, 24)

    def __post_init__(self) -> None:
        assert self.nx > 0 and self.ny > 0 and self.nz > 0, \
            "Element counts must be positive integers."
        self._build_mesh()

    def _build_mesh(self) -> None:
        """
        Build node coordinates, element connectivity, and DOF map.

        Node ordering within each hex element follows the standard
        isoparametric convention (Zienkiewicz & Taylor, Vol 1):
        nodes 0-3 on bottom face (z=0), nodes 4-7 on top face (z=1),
        both in counter-clockwise order when viewed from outside.

        DOF map: for element e, dof_map[e] contains the 24 global DOF
        indices [u1,v1,w1, u2,v2,w2, ..., u8,v8,w8] for the 8 nodes.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        nnx, nny, nnz = nx + 1, ny + 1, nz + 1

        self.n_elem  = nx * ny * nz
        self.n_nodes = nnx * nny * nnz
        self.n_dof   = 3 * self.n_nodes

        # Node coordinates — shape (n_nodes, 3)
        xs = np.linspace(0, self.lx, nnx)
        ys = np.linspace(0, self.ly, nny)
        zs = np.linspace(0, self.lz, nnz)
        # meshgrid with z-fastest (C-order) indexing
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        self.coords = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=1
        )  # shape (n_nodes, 3)

        # Element connectivity — shape (n_elem, 8)
        # Node index for element (i,j,k):
        #   bottom face: n(i,j,k), n(i+1,j,k), n(i+1,j+1,k), n(i,j+1,k)
        #   top face:    n(i,j,k+1), ...(same pattern)
        def nid(i, j, k):
            return i * nny * nnz + j * nnz + k

        ei = np.arange(nx)
        ej = np.arange(ny)
        ek = np.arange(nz)
        ii, jj, kk = np.meshgrid(ei, ej, ek, indexing='ij')
        ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

        self.conn = np.stack([
            nid(ii,   jj,   kk),
            nid(ii+1, jj,   kk),
            nid(ii+1, jj+1, kk),
            nid(ii,   jj+1, kk),
            nid(ii,   jj,   kk+1),
            nid(ii+1, jj,   kk+1),
            nid(ii+1, jj+1, kk+1),
            nid(ii,   jj+1, kk+1),
        ], axis=1).astype(np.int32)  # shape (n_elem, 8)

        # DOF map — shape (n_elem, 24)
        # For each node n in element, DOFs are [3n, 3n+1, 3n+2]
        node_dofs = np.stack([
            3 * self.conn,
            3 * self.conn + 1,
            3 * self.conn + 2,
        ], axis=2)  # shape (n_elem, 8, 3)
        self.dof_map = node_dofs.reshape(self.n_elem, 24)

    def element_centres(self) -> np.ndarray:
        """
        Return element centroid coordinates.

        Returns
        -------
        np.ndarray, shape (n_elem, 3)
        """
        return self.coords[self.conn].mean(axis=1)

    def neighbour_map(self, radius: int = 1) -> list[np.ndarray]:
        """
        Return list of neighbour element index arrays for the density filter.

        For element e, neighbour_map[e] contains indices of all elements
        within Chebyshev distance <= radius (including e itself).
        Used by the sensitivity filter in chunk 3.

        Parameters
        ----------
        radius : int
            Filter radius in elements. Default 1.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        idx = np.arange(self.n_elem).reshape(nx, ny, nz)
        neighbours = []
        for e in range(self.n_elem):
            i, j, k = np.unravel_index(e, (nx, ny, nz))
            i0, i1 = max(0, i-radius), min(nx, i+radius+1)
            j0, j1 = max(0, j-radius), min(ny, j+radius+1)
            k0, k1 = max(0, k-radius), min(nz, k+radius+1)
            neighbours.append(idx[i0:i1, j0:j1, k0:k1].ravel())
        return neighbours

    def summary(self) -> str:
        return (
            f"VoxelMesh: {self.nx}\u00D7{self.ny}\u00D7{self.nz}  "
            f"| elements={self.n_elem:,}  "
            f"| nodes={self.n_nodes:,}  "
            f"| DOFs={self.n_dof:,}"
        )


# ---------------------------------------------------------------------------
# FEAssembler
# ---------------------------------------------------------------------------

class FEAssembler:
    """
    Assemble the global stiffness matrix K from element stiffness matrices.

    Assembly strategy (Decision 4):
        1. Compute all Ke^0 (full-density element stiffness) once at init.
        2. On each SIMP call, scale by rho_e^p and scatter into COO format.
        3. Convert COO -> CSC once for the linear solve (spsolve).
        4. Convert COO -> CSR for matrix-vector products (sensitivity).

    Element type: trilinear hexahedral (Q8), 8 nodes, 24 DOFs per element.
    Numerical integration: 2x2x2 Gauss quadrature (exact for bilinear fields).

    Parameters
    ----------
    mesh : VoxelMesh
        The mesh to assemble over.
    material : MaterialModel
        Provides C0 (full-density Voigt stiffness matrix).
    """

    # 2x2x2 Gauss points and weights for [-1,1]^3
    _GP  = np.array([-1, 1]) / np.sqrt(3)
    _GW  = np.array([1.0, 1.0])

    def __init__(self, mesh: VoxelMesh, material: MaterialModel) -> None:
        self.mesh     = mesh
        self.material = material
        self._Ke0     = self._compute_Ke0()  # shape (24, 24) — same for all elements

        # Pre-build COO index arrays (fixed for structured mesh)
        dof = mesh.dof_map                  # (n_elem, 24)
        n   = mesh.n_elem
        # Row and column indices for all element DOF pairs
        self._rows = np.repeat(dof, 24, axis=1).ravel()   # (n_elem*576,)
        self._cols = np.tile(dof, (1, 24)).ravel()         # (n_elem*576,)

    def _compute_Ke0(self) -> np.ndarray:
        """
        Compute the full-density element stiffness matrix Ke0.

        For a regular hex mesh all elements are identical (same Jacobian),
        so Ke0 is computed once and reused.

        Ke0 = integral_Omega B^T C0 B dOmega
            ≈ sum_q w_q * B(xi_q)^T * C0 * B(xi_q) * det(J)

        where B is the 6x24 strain-displacement matrix evaluated at
        each Gauss point xi_q, and J is the Jacobian of the isoparametric
        mapping.

        Returns
        -------
        np.ndarray, shape (24, 24)
        """
        mesh = self.mesh
        C0   = self.material.C0
        h    = np.array([
            mesh.lx / mesh.nx,
            mesh.ly / mesh.ny,
            mesh.lz / mesh.nz
        ])  # element dimensions

        Ke0 = np.zeros((24, 24))
        gp, gw = self._GP, self._GW

        for i, xi in enumerate(gp):
            for j, eta in enumerate(gp):
                for k, zeta in enumerate(gp):
                    B, detJ = self._strain_displacement(xi, eta, zeta, h)
                    w = gw[i] * gw[j] * gw[k]
                    Ke0 += w * detJ * (B.T @ C0 @ B)

        return Ke0

    def _strain_displacement(
        self, xi: float, eta: float, zeta: float, h: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Compute the 6x24 strain-displacement matrix B and Jacobian det
        at a Gauss point (xi, eta, zeta) in [-1,1]^3.

        Shape functions for trilinear hex (standard isoparametric):
            N_i(xi,eta,zeta) = (1 + xi_i*xi)(1 + eta_i*eta)(1 + zeta_i*zeta)/8

        For a regular axis-aligned element with dimensions hx x hy x hz,
        the Jacobian is diagonal: J = diag(hx/2, hy/2, hz/2), detJ = hx*hy*hz/8.

        Returns
        -------
        B : np.ndarray, shape (6, 24)
        detJ : float
        """
        hx, hy, hz = h

        # Local node coordinates in [-1,1]^3 (standard hex ordering)
        xi_n   = np.array([-1,  1,  1, -1, -1,  1,  1, -1], dtype=float)
        eta_n  = np.array([-1, -1,  1,  1, -1, -1,  1,  1], dtype=float)
        zeta_n = np.array([-1, -1, -1, -1,  1,  1,  1,  1], dtype=float)

        # Shape function derivatives w.r.t. local coords
        dN_dxi   = xi_n   * (1 + eta_n*eta)   * (1 + zeta_n*zeta) / 8
        dN_deta  = eta_n  * (1 + xi_n*xi)     * (1 + zeta_n*zeta) / 8
        dN_dzeta = zeta_n * (1 + xi_n*xi)     * (1 + eta_n*eta)   / 8

        # Jacobian (diagonal for regular mesh)
        detJ = hx * hy * hz / 8.0

        # Convert to physical coords: dN/dx = dN/dxi * 2/hx (chain rule)
        dN_dx = dN_dxi   * 2.0 / hx
        dN_dy = dN_deta  * 2.0 / hy
        dN_dz = dN_dzeta * 2.0 / hz

        # Assemble B matrix (Voigt convention: [exx, eyy, ezz, gyz, gxz, gxy])
        B = np.zeros((6, 24))
        for n in range(8):
            col = 3 * n
            B[0, col]   = dN_dx[n]   # exx = du/dx
            B[1, col+1] = dN_dy[n]   # eyy = dv/dy
            B[2, col+2] = dN_dz[n]   # ezz = dw/dz
            B[3, col+1] = dN_dz[n]   # gyz = dv/dz + dw/dy
            B[3, col+2] = dN_dy[n]
            B[4, col]   = dN_dz[n]   # gxz = du/dz + dw/dx
            B[4, col+2] = dN_dx[n]
            B[5, col]   = dN_dy[n]   # gxy = du/dy + dv/dx
            B[5, col+1] = dN_dx[n]

        return B, detJ

    def assemble_K(
        self,
        rho: np.ndarray,
        p:   float = 3.0,
        E_min_factor: float = 1e-9
    ) -> sp.csc_matrix:
        """
        Assemble the global stiffness matrix K for a given density field.

        K = sum_e (E_min + rho_e^p * (1-E_min)) * Ke0

        Assembly: COO scatter -> CSC conversion (Decision 4).

        Parameters
        ----------
        rho : np.ndarray, shape (n_elem,)
            Element densities in [0, 1].
        p : float
            SIMP penalisation exponent. Default 3.
        E_min_factor : float
            Void stiffness floor. Default 1e-9.

        Returns
        -------
        K : scipy.sparse.csc_matrix, shape (n_dof, n_dof)
        """
        assert rho.shape == (self.mesh.n_elem,), \
            f"rho must have shape ({self.mesh.n_elem},), got {rho.shape}"

        E_min  = E_min_factor
        scales = E_min + rho ** p * (1.0 - E_min)  # (n_elem,)

        # Scale Ke0 for each element and flatten to COO data
        # Ke0 is (24,24), scales is (n_elem,) -> (n_elem, 24, 24)
        Ke_all = scales[:, np.newaxis, np.newaxis] * self.Ke0  # broadcast
        data   = Ke_all.ravel()  # (n_elem * 576,)

        n_dof = self.mesh.n_dof
        K_coo = sp.coo_matrix(
            (data, (self._rows, self._cols)),
            shape=(n_dof, n_dof)
        )
        return K_coo.tocsc()

    @property
    def Ke0(self) -> np.ndarray:
        """Full-density element stiffness matrix, shape (24, 24)."""
        return self._Ke0

    def summary(self) -> str:
        n_dof = self.mesh.n_dof
        nnz_est = 81 * n_dof
        mem_mb  = nnz_est * 8 / 1e6
        return (
            f"FEAssembler: {self.mesh.nx}\u00D7{self.mesh.ny}\u00D7{self.mesh.nz}  "
            f"| K size={n_dof:,}\u00D7{n_dof:,}  "
            f"| nnz\u2248{nnz_est:,}  "
            f"| mem\u2248{mem_mb:.0f} MB"
        )
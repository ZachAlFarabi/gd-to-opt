"""
verify.py — Chunk 4, Decisions 17+18: verification FEA and mechanical report

Two solvers:
    ThermalAssembler — assembles K_T for steady-state heat conduction
                       scalar DOF (temperature), same Q8 hex elements
    verify()         — runs structural + optional thermal verification FEA
                       on the binary density field from SIMPResult

Thermal mode B (Decision 17):
    1. Solve K_T T = f_T  (heat conduction)
    2. Compute thermal strain eps_th = alpha*(T_e - T_ref)*[1,1,1,0,0,0]
    3. Add thermal load f_th = sum_e Ke0 @ eps_th_e to structural RHS
    4. Solve structural K_ff u_f = f_mech_f + f_th_f
    5. Compute von Mises stress, safety factor, gap ratio

New material constants added to presets:
    k_thermal (W/m/K) — thermal conductivity
    alpha_CTE (1/K)   — coefficient of thermal expansion
    T_ref (°C)        — reference temperature
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from typing import NamedTuple

from gdto.mesh_material import VoxelMesh, MaterialModel, FEAssembler, BoundaryConditions, ProblemData
from gdto.simp import SIMPResult


# ---------------------------------------------------------------------------
# Thermal material constants (not in user CSV — from literature)
# ---------------------------------------------------------------------------

THERMAL_CONSTANTS = {
    "Ti64":     {"k": 6.7,   "alpha": 8.6e-6,  "T_ref": 20.0},
    "AlSi10Mg": {"k": 130.0, "alpha": 21.0e-6, "T_ref": 20.0},
    "316L":     {"k": 16.3,  "alpha": 16.0e-6, "T_ref": 20.0},
}


# ---------------------------------------------------------------------------
# VerifyResult
# ---------------------------------------------------------------------------

class VerifyResult(NamedTuple):
    """
    Complete verification result returned to the GD loop and frontend.

    Structural fields:
        compliance_real    — C on binary {0,1} geometry  [N·m]
        compliance_simp    — C from SIMPResult (penalised) [N·m]
        gap_ratio          — (C_real - C_simp) / C_simp
        max_von_mises_mpa  — max von Mises stress over solid elements [MPa]
        min_safety_factor  — sigma_yield / max_von_mises
        mass_kg            — mass from binary density field [kg]
        volume_fraction    — mean(rho_binary)

    Thermal fields (None if thermal solve not run):
        max_temp_c         — maximum temperature [°C]
        max_thermal_stress_mpa — max von Mises from thermal loading only

    Geometry fields:
        stl_info           — dict from reconstruct.reconstruct_stl

    Metadata:
        material           — material preset name
        wall_time_s        — total verification time [s]
        report             — dict suitable for JSON serialisation
    """
    # Structural
    compliance_real:       float
    compliance_simp:       float
    gap_ratio:             float
    max_von_mises_mpa:     float
    min_safety_factor:     float
    mass_kg:               float
    volume_fraction:       float

    # Thermal (None if not run)
    max_temp_c:            float | None
    max_thermal_stress_mpa: float | None

    # Geometry
    stl_info:              dict | None

    # Metadata
    material:              str
    wall_time_s:           float
    report:                dict

    # Per-element fields for frontend heatmaps
    stress_field_mpa:      np.ndarray | None = None
    temp_field_c:          np.ndarray | None = None


# ---------------------------------------------------------------------------
# ThermalAssembler
# ---------------------------------------------------------------------------

class ThermalAssembler:
    """
    Assembles the thermal stiffness matrix K_T for steady-state
    heat conduction on the Q8 hex mesh.

    K_T[i,j] = integral_Omega k * grad(N_i) . grad(N_j) dOmega

    Scalar DOF per node — K_T is (n_nodes, n_nodes), much smaller
    than the structural K which is (3*n_nodes, 3*n_nodes).

    The same shape functions and Gauss quadrature as FEAssembler.
    """

    _GP = np.array([-1, 1]) / np.sqrt(3)
    _GW = np.array([1.0, 1.0])

    def __init__(self, mesh: VoxelMesh, k_thermal: float) -> None:
        self.mesh      = mesh
        self.k         = k_thermal
        self._KTe0     = self._compute_KTe0()

        # Pre-build COO indices for scalar DOF
        # Each element has 8 nodes → 64 (row, col) pairs
        conn = mesh.conn           # (n_elem, 8)
        n    = mesh.n_elem
        self._rows = np.repeat(conn, 8, axis=1).ravel()   # (n_elem*64,)
        self._cols = np.tile(conn, (1, 8)).ravel()         # (n_elem*64,)

    def _compute_KTe0(self) -> np.ndarray:
        """
        Compute full-density thermal element stiffness matrix KTe0.
        Shape: (8, 8) — scalar DOF per node.
        """
        mesh = self.mesh
        k    = self.k
        hx   = mesh.lx / mesh.nx
        hy   = mesh.ly / mesh.ny
        hz   = mesh.lz / mesh.nz
        h    = np.array([hx, hy, hz])

        KTe0 = np.zeros((8, 8))
        gp, gw = self._GP, self._GW

        xi_n   = np.array([-1, 1,  1, -1, -1,  1,  1, -1], dtype=float)
        eta_n  = np.array([-1,-1,  1,  1, -1, -1,  1,  1], dtype=float)
        zeta_n = np.array([-1,-1, -1, -1,  1,  1,  1,  1], dtype=float)

        for i, xi in enumerate(gp):
            for j, eta in enumerate(gp):
                for kk, zeta in enumerate(gp):
                    dN_dxi   = xi_n   * (1+eta_n*eta)   * (1+zeta_n*zeta) / 8
                    dN_deta  = eta_n  * (1+xi_n*xi)     * (1+zeta_n*zeta) / 8
                    dN_dzeta = zeta_n * (1+xi_n*xi)     * (1+eta_n*eta)   / 8

                    detJ = hx * hy * hz / 8.0
                    dN_dx = dN_dxi   * 2.0 / hx
                    dN_dy = dN_deta  * 2.0 / hy
                    dN_dz = dN_dzeta * 2.0 / hz

                    # grad_N: (3, 8)
                    grad_N = np.stack([dN_dx, dN_dy, dN_dz], axis=0)
                    w = gw[i] * gw[j] * gw[kk]
                    KTe0 += w * detJ * k * (grad_N.T @ grad_N)

        return KTe0

    def assemble_KT(
        self, rho_binary: np.ndarray, linear: bool = False
    ) -> sp.csc_matrix:
        """
        Assemble global thermal stiffness matrix K_T.

        Parameters
        ----------
        rho_binary : element densities (binary in verify, continuous in SIMP)
        linear     : if True, use continuous linear interpolation of conductivity
                     (for SIMP loop); if False, use binary threshold (for verify)

        Returns
        -------
        K_T : sp.csc_matrix, shape (n_nodes, n_nodes)
        """
        mesh    = self.mesh
        # Void elements represent air/insulation, not vacuum.
        # k_air ≈ 0.026 W/mK; void ratio = max(k_air/k_solid, 0.002).
        k_void_ratio = max(0.026 / self.k, 0.002)
        if linear:
            # Continuous interpolation — avoids all-void problem in early SIMP iters
            scales = k_void_ratio + rho_binary * (1.0 - k_void_ratio)
        else:
            scales = np.where(rho_binary > 0.5, 1.0, k_void_ratio)

        KTe_all = scales[:, np.newaxis, np.newaxis] * self._KTe0[np.newaxis]
        data    = KTe_all.ravel()

        K_T_coo = sp.coo_matrix(
            (data, (self._rows, self._cols)),
            shape=(mesh.n_nodes, mesh.n_nodes)
        )
        return K_T_coo.tocsc()

    def build_thermal_load(
        self,
        flux_faces:  dict[str, float],
        temp_faces:  dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build thermal load vector f_T and identify fixed temperature DOFs.

        Parameters
        ----------
        flux_faces : dict face→flux  e.g. {"zmin": 5000.0}  W/m²
        temp_faces : dict face→temp  e.g. {"zmax": 20.0}    °C

        Returns
        -------
        f_T        : np.ndarray, shape (n_nodes,)
        fixed_dofs : np.ndarray of node indices with prescribed T
        fixed_vals : np.ndarray of prescribed temperatures [°C]
        """
        mesh  = self.mesh
        f_T   = np.zeros(mesh.n_nodes)

        fixed_dofs_list = []
        fixed_vals_list = []

        # Heat flux faces → Neumann BC
        for face, flux in flux_faces.items():
            node_ids = self._face_node_ids(face)
            # Distribute flux over face nodes uniformly
            # face area = product of the two non-normal dimensions
            area = self._face_area(face)
            f_T[node_ids] += flux * area / len(node_ids)

        # Fixed temperature faces → Dirichlet BC
        for face, temp in temp_faces.items():
            node_ids = self._face_node_ids(face)
            fixed_dofs_list.extend(node_ids.tolist())
            fixed_vals_list.extend([float(temp)] * len(node_ids))

        fixed_dofs = np.array(fixed_dofs_list, dtype=np.int32)
        fixed_vals = np.array(fixed_vals_list, dtype=np.float64)
        return f_T, fixed_dofs, fixed_vals

    def _face_node_ids(self, face: str) -> np.ndarray:
        """Return node indices on the named face (same logic as BoundaryConditions)."""
        mesh = self.mesh
        nnx, nny, nnz = mesh.nx+1, mesh.ny+1, mesh.nz+1
        ii, jj, kk = np.arange(nnx), np.arange(nny), np.arange(nnz)
        face_map = {
            'xmin': (np.array([0]),     jj, kk),
            'xmax': (np.array([nnx-1]), jj, kk),
            'ymin': (ii, np.array([0]),     kk),
            'ymax': (ii, np.array([nny-1]), kk),
            'zmin': (ii, jj, np.array([0])),
            'zmax': (ii, jj, np.array([nnz-1])),
        }
        if face not in face_map:
            raise ValueError(f"Unknown face '{face}'")
        fi, fj, fk = face_map[face]
        gi, gj, gk = np.meshgrid(fi, fj, fk, indexing='ij')
        return (gi.ravel() * nny * nnz + gj.ravel() * nnz + gk.ravel()).astype(np.int32)

    def _face_area(self, face: str) -> float:
        """Physical area of a named face [m²]."""
        mesh = self.mesh
        areas = {
            'xmin': mesh.ly * mesh.lz, 'xmax': mesh.ly * mesh.lz,
            'ymin': mesh.lx * mesh.lz, 'ymax': mesh.lx * mesh.lz,
            'zmin': mesh.lx * mesh.ly, 'zmax': mesh.lx * mesh.ly,
        }
        return areas[face]

    def solve_temperature(
        self,
        rho_binary:  np.ndarray,
        flux_faces:  dict[str, float],
        temp_faces:  dict[str, float],
        linear:      bool = False,
    ) -> np.ndarray:
        """
        Solve K_T T = f_T for the temperature field.

        Parameters
        ----------
        linear : passed through to assemble_KT — use True in SIMP loop

        Returns
        -------
        T : np.ndarray, shape (n_nodes,) — temperature at each node [°C]
        """
        mesh = self.mesh
        K_T  = self.assemble_KT(rho_binary, linear=linear)
        f_T, fixed_dofs, fixed_vals = self.build_thermal_load(
            flux_faces, temp_faces
        )

        # If no Dirichlet BC provided, K_T is singular (no temperature reference).
        # Auto-pin one corner node to T_ref to anchor the solution, but warn —
        # the resulting temperature field is mathematically stable but physically
        # meaningless without a real heat-sink face.
        if len(fixed_dofs) == 0:
            import warnings
            warnings.warn(
                "No fixed temperature BC provided. K_T is singular — anchoring "
                "corner node to T_ref=20°C. Temperature field will not be "
                "physically meaningful. Add a fixed temperature face (heat sink) "
                "opposite to the flux face for a valid thermal result.",
                UserWarning, stacklevel=3,
            )
            fixed_dofs = np.array([0], dtype=np.int32)
            fixed_vals = np.array([20.0], dtype=np.float64)

        all_dofs  = np.arange(mesh.n_nodes)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        if len(fixed_dofs) > 0:
            f_T[free_dofs] -= np.array(
                K_T[np.ix_(free_dofs, fixed_dofs)] @ fixed_vals
            ).ravel()

        K_ff = K_T[np.ix_(free_dofs, free_dofs)].tocsc()
        f_f  = f_T[free_dofs]

        T_free = spla.spsolve(K_ff, f_f)

        T = np.zeros(mesh.n_nodes)
        T[free_dofs]  = T_free
        T[fixed_dofs] = fixed_vals
        return T


# ---------------------------------------------------------------------------
# verify() — main entry point
# ---------------------------------------------------------------------------

def verify(
    simp_result:    SIMPResult,
    problem:        ProblemData,
    stl_path:       str | None = None,
    flux_faces:     dict[str, float] | None = None,
    temp_faces:     dict[str, float] | None = None,
    T_field_precomputed: np.ndarray | None = None,
    yield_strength_mpa: float | None = None,
) -> VerifyResult:
    """
    Run verification FEA on the binary density field from SIMPResult.

    Structural verification (always):
        rho_binary = threshold(rho_filtered, 0.5)
        Solve K(rho_binary) u = f_mech + f_thermal
        Compute compliance, von Mises stress, safety factor

    Thermal (Mode B, if flux_faces or temp_faces provided):
        1. Solve heat conduction K_T T = f_T
        2. Compute thermal load f_thermal from temperature field
        3. Add to structural RHS

    Parameters
    ----------
    simp_result : SIMPResult from run_simp()
    problem     : ProblemData from build_problem()
    stl_path    : path to reconstructed STL (for report info only)
    flux_faces  : dict face→W/m² for heat flux BCs
    temp_faces  : dict face→°C for fixed temperature BCs
    yield_strength_mpa : override material yield strength [MPa]

    Returns
    -------
    VerifyResult
    """
    t_start  = time.perf_counter()
    mesh     = problem.mesh
    mat      = problem.material
    asm      = problem.assembler
    bc       = problem.bc

    flux_faces = flux_faces or {}
    temp_faces = temp_faces or {}

    # Binary density field — no SIMP penalisation
    rho_binary = np.where(simp_result.rho > 0.5, 1.0, 0.001)

    # ── Thermal solve (Mode B) ─────────────────────────────────────────
    f_thermal = np.zeros(mesh.n_dof)
    T_field   = None
    max_temp  = None
    max_thermal_stress_mpa = None

    run_thermal = len(flux_faces) > 0 or len(temp_faces) > 0

    if run_thermal:
        tc = THERMAL_CONSTANTS.get(mat.name, THERMAL_CONSTANTS["Ti64"])
        k_th    = tc["k"]
        alpha   = tc["alpha"]
        T_ref   = tc["T_ref"]

        if T_field_precomputed is not None:
            # Reuse the SIMP-converged temperature field — avoids double solve
            # and keeps thermal loading consistent between optimisation and verify
            T_field = T_field_precomputed
        else:
            th_asm  = ThermalAssembler(mesh, k_th)
            T_field = th_asm.solve_temperature(rho_binary, flux_faces, temp_faces)
        max_temp = float(T_field.max())

        # Element centroid temperatures
        T_elem = T_field[mesh.conn].mean(axis=1)   # (n_elem,)
        dT     = T_elem - T_ref                     # temperature rise

        # Thermal strain vector per element: alpha*dT*[1,1,1,0,0,0]
        eps_th = np.zeros((mesh.n_elem, 6))
        eps_th[:, :3] = (alpha * dT)[:, np.newaxis]

        # Thermal load: f_th_e = Ke0 @ (C0 @ eps_th_e) integrated over element
        # For uniform strain in element: f_th_e = v_e * B_c^T C0 eps_th_e
        # Approximation: use Ke0 @ eps_th via element DOF mapping
        # Full vectorised: (n_elem, 24) = (n_elem, 6) @ C0.T -> (n_elem, 6) then B^T
        # Practical: compute per element using Ke0 and thermal strain
        v_elem  = (mesh.lx * mesh.ly * mesh.lz) / mesh.n_elem
        h_vec   = np.array([mesh.lx/mesh.nx, mesh.ly/mesh.ny, mesh.lz/mesh.nz])
        B_c, _  = asm._strain_displacement(0, 0, 0, h_vec)  # (6, 24)

        # Scale thermal stress by binary mask — void elements must NOT contribute
        # spurious thermal forces (without this, SF=2685 and 10× stress overestimate)
        scales_verify = np.where(rho_binary > 0.5, 1.0, 0.0)
        sig_th = eps_th @ mat.C0.T * scales_verify[:, np.newaxis]  # (n_elem, 6)

        f_th_elem  = (sig_th @ B_c).reshape(mesh.n_elem, 24) * v_elem  # (n_elem, 24)

        # Scatter to global
        for e in range(mesh.n_elem):
            f_thermal[mesh.dof_map[e]] += f_th_elem[e]

        # Thermal-only von Mises for report
        Ue_th = np.zeros((mesh.n_elem, 24))  # placeholder — computed below

    # ── Structural solve on binary mesh ───────────────────────────────
    K      = asm.assemble_K(rho_binary, p=1.0, E_min_factor=0.001)
    K_ff, f_mech_f = bc.apply(K)

    # Add thermal load to mechanical load
    f_total_f = f_mech_f + f_thermal[bc.free_dofs]

    # If total load is negligible (no mechanical + no thermal), skip the stress
    # solve entirely.  The -1e-6 N K-conditioning fallback would otherwise produce
    # numerical noise that the percentile colormap stretches to full rainbow.
    no_load = float(np.linalg.norm(f_total_f)) < 1e-4

    solid_mask = rho_binary > 0.5
    vm_full    = np.zeros(mesh.n_elem, dtype=np.float32)
    max_vm_mpa = 0.0
    compliance_real = 0.0

    if no_load:
        u_full = np.zeros(mesh.n_dof)
    else:
        u_free = spla.spsolve(K_ff, f_total_f)
        u_full = bc.expand(u_free)

        # ── Compliance ────────────────────────────────────────────────────
        compliance_real = float(np.dot(bc.f, u_full))

        # ── Stress field (solid elements only) ────────────────────────────
        h_vec = np.array([mesh.lx/mesh.nx, mesh.ly/mesh.ny, mesh.lz/mesh.nz])
        B_c, _ = asm._strain_displacement(0.0, 0.0, 0.0, h_vec)  # (6, 24)

        Ue_solid  = u_full[mesh.dof_map[solid_mask]]     # (n_solid, 24)
        eps_solid = Ue_solid @ B_c.T                      # (n_solid, 6)

        # Subtract thermal strain for solid elements if thermal solve ran
        if run_thermal:
            eps_solid -= eps_th[solid_mask]

        sig_solid = eps_solid @ mat.C0.T                  # (n_solid, 6)

        s   = sig_solid
        dxx = s[:,0] - s[:,1]
        dyy = s[:,1] - s[:,2]
        dzz = s[:,2] - s[:,0]
        vm  = np.sqrt(0.5*(dxx**2 + dyy**2 + dzz**2)
                      + 3*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2))

        max_vm_mpa = float(vm.max() / 1e6) if len(vm) > 0 else 0.0
        vm_full[solid_mask] = (vm / 1e6).astype(np.float32)

    # Per-element temperature field (average node temps per element)
    temp_elem_full = None
    if run_thermal and T_field is not None:
        temp_elem_full = T_field[mesh.conn].mean(axis=1).astype(np.float32)

    # Safety factor
    if yield_strength_mpa is None:
        # Default: use XY yield from CSV (conservative, lower value)
        yield_defaults = {"Ti64": 1125.0, "AlSi10Mg": 265.0, "316L": 547.0}
        yield_strength_mpa = yield_defaults.get(mat.name, 300.0)

    sf = float(yield_strength_mpa / max_vm_mpa) if max_vm_mpa > 0 else float('inf')

    # ── Mass and volume fraction ───────────────────────────────────────
    v_elem     = (mesh.lx * mesh.ly * mesh.lz) / mesh.n_elem
    mass_kg    = float(mat.rho * v_elem * rho_binary.sum())
    vol_frac   = float(rho_binary[rho_binary > 0.5].shape[0] / mesh.n_elem)

    # ── Gap ratio ─────────────────────────────────────────────────────
    # Signed improvement: positive = SIMP was conservative (binary is stiffer, good)
    #                     negative = SIMP was optimistic (unusual, means well-converged)
    gap = (simp_result.compliance - compliance_real) / max(abs(simp_result.compliance), 1e-30)
    gap = float(np.clip(gap, -1.0, 100.0))

    wall_time = time.perf_counter() - t_start

    # ── STL info ──────────────────────────────────────────────────────
    stl_info = None
    if stl_path is not None:
        from pathlib import Path
        stl_info = {"stl_path": str(stl_path)}

    # ── Build report dict ─────────────────────────────────────────────
    report = {
        "compliance_simp_Nm":   round(simp_result.compliance, 6),
        "compliance_real_Nm":   round(compliance_real, 6),
        "gap_ratio_pct":        round(gap * 100, 2),
        "gap_label":            "SIMP vs real (+ = conservative)",
        "mass_kg":              round(mass_kg, 4),
        "volume_fraction":      round(vol_frac, 4),
        "support_volume_m3":    round(simp_result.support_volume, 6),
        "max_von_mises_mpa":    round(max_vm_mpa, 2),
        "min_safety_factor":    None if not np.isfinite(sf) else round(sf, 3),
        "yield_strength_mpa":   yield_strength_mpa,
        "material":             mat.name,
        "mesh":                 {"nx": mesh.nx, "ny": mesh.ny, "nz": mesh.nz},
        "thermal_active":       run_thermal,
        "max_temp_c":           round(max_temp, 2) if max_temp is not None else None,
        "stress_label":         "Von Mises (thermoelastic)" if run_thermal else "Von Mises (mechanical)",
        "wall_time_s":          round(wall_time, 2),
    }
    if stl_info:
        report["stl"] = stl_info

    return VerifyResult(
        compliance_real        = compliance_real,
        compliance_simp        = simp_result.compliance,
        gap_ratio              = float(gap),
        max_von_mises_mpa      = max_vm_mpa,
        min_safety_factor      = sf,
        mass_kg                = mass_kg,
        volume_fraction        = vol_frac,
        max_temp_c             = max_temp,
        max_thermal_stress_mpa = max_thermal_stress_mpa,
        stl_info               = stl_info,
        material               = mat.name,
        wall_time_s            = wall_time,
        report                 = report,
        stress_field_mpa       = vm_full,
        temp_field_c           = temp_elem_full,
    )
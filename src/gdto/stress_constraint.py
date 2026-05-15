"""
stress_constraint.py — Chunk 6: stress-constrained SIMP

Implements P-norm stress aggregation and adjoint sensitivity
for stress-constrained topology optimisation.

Decisions implemented:
    D28 — P-norm aggregation: tilde_sigma = (sum(rho^q * sigma_vM / sigma_y)^P)^(1/P)
    D29 — Full adjoint sensitivity via second linear solve
    D30 — Separate stress penalisation exponent q=0.5
    D31 — Augmented Lagrangian penalty added to compliance objective
    D32 — Safety factor input, SF=0 disables constraint (backward compat)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from gdto.mesh_material import VoxelMesh, FEAssembler, MaterialModel


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class StressConfig:
    """
    Stress constraint configuration.

    safety_factor   : target SF = sigma_yield / sigma_limit.
                      SF=0 disables the stress constraint entirely.
    p_norm          : P-norm aggregation exponent (D28). Higher = tighter.
    q_stress        : SIMP penalisation for stress (D30). q < p to avoid singularity.
    penalty_weight  : augmented Lagrangian weight mu (D31).
    penalty_update  : multiply mu by this each iteration when constraint violated.
    """
    safety_factor:  float = 0.0     # 0 = disabled (backward compat)
    p_norm:         int   = 6       # P-norm exponent
    q_stress:       float = 0.5     # stress SIMP exponent
    penalty_weight: float = 1.0     # augmented Lagrangian mu
    penalty_update: float = 1.5     # mu growth rate on violation


# ── Von Mises stress computation ──────────────────────────────────────────────

def compute_element_stress(
    u_full:   np.ndarray,
    mesh:     VoxelMesh,
    asm:      FEAssembler,
    material: MaterialModel,
) -> np.ndarray:
    """
    Compute von Mises stress at element centroids.

    Parameters
    ----------
    u_full   : (n_dof,) displacement vector
    mesh     : VoxelMesh
    asm      : FEAssembler
    material : MaterialModel

    Returns
    -------
    sigma_vm : (n_elem,) von Mises stress [Pa]
    """
    h_vec = np.array([mesh.lx / mesh.nx,
                      mesh.ly / mesh.ny,
                      mesh.lz / mesh.nz])
    B_c, _ = asm._strain_displacement(0.0, 0.0, 0.0, h_vec)  # (6, 24)

    # Element displacements: (n_elem, 24)
    Ue  = u_full[mesh.dof_map]

    # Strain at centroid: (n_elem, 6)
    eps = Ue @ B_c.T

    # Stress: (n_elem, 6)
    sig = eps @ material.C0.T

    # Von Mises: sqrt(0.5 * [(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2]
    #                   + 3 * [tyz^2 + txz^2 + txy^2])
    dxx = sig[:, 0] - sig[:, 1]
    dyy = sig[:, 1] - sig[:, 2]
    dzz = sig[:, 2] - sig[:, 0]
    sigma_vm = np.sqrt(
        0.5 * (dxx**2 + dyy**2 + dzz**2) +
        3.0 * (sig[:, 3]**2 + sig[:, 4]**2 + sig[:, 5]**2)
    )
    return sigma_vm


# ── P-norm aggregation ────────────────────────────────────────────────────────

def p_norm_stress(
    sigma_vm:    np.ndarray,
    rho:         np.ndarray,
    sigma_yield: float,
    P:           int   = 6,
    q:           float = 0.5,
) -> float:
    """
    Compute P-norm stress aggregation (D28).

    The P-norm approaches max(sigma_vm) as P → ∞.
    Stress is penalised by rho^q to avoid singularities in void (D30).

    tilde_sigma = (sum_e (rho_e^q * sigma_vm_e / sigma_yield)^P)^(1/P)

    Parameters
    ----------
    sigma_vm    : (n_elem,) von Mises stress [Pa]
    rho         : (n_elem,) filtered density
    sigma_yield : yield strength [Pa]
    P           : P-norm exponent
    q           : stress SIMP exponent

    Returns
    -------
    float : normalised P-norm stress (constraint: <= 1)
    """
    s_norm = (rho ** q) * (sigma_vm / sigma_yield)

    s_max  = s_norm.max()
    if s_max < 1e-30:
        return 0.0

    # Normalise to avoid overflow in large P
    s_rel  = s_norm / s_max
    p_sum  = np.sum(s_rel ** P)
    tilde  = s_max * (p_sum ** (1.0 / P))
    return float(tilde)


# ── P-norm sensitivity ────────────────────────────────────────────────────────

def p_norm_sensitivity_direct(
    sigma_vm:    np.ndarray,
    rho:         np.ndarray,
    sigma_yield: float,
    P:           int   = 6,
    q:           float = 0.5,
) -> np.ndarray:
    """
    Direct term of d(tilde_sigma)/d(rho_e) — the density scaling term.

    This is the derivative with respect to rho through the rho^q factor only.
    The adjoint term (through displacement) is computed separately.

    Parameters
    ----------
    sigma_vm    : (n_elem,) von Mises stress [Pa]
    rho         : (n_elem,) filtered density
    sigma_yield : yield strength [Pa]
    P           : P-norm exponent
    q           : stress SIMP exponent

    Returns
    -------
    ds_drho : (n_elem,) direct sensitivity
    """
    s_norm = (rho ** q) * (sigma_vm / sigma_yield)
    s_max  = s_norm.max()
    if s_max < 1e-30:
        return np.zeros_like(rho)

    s_rel  = s_norm / s_max
    p_sum  = np.sum(s_rel ** P)
    tilde  = s_max * (p_sum ** (1.0 / P))

    if tilde < 1e-30:
        return np.zeros_like(rho)

    # d(tilde)/d(s_norm_e) = tilde^(1-P) * s_norm_e^(P-1)
    dt_ds = (tilde ** (1 - P)) * (s_norm ** (P - 1))

    # d(s_norm_e)/d(rho_e) = q * rho_e^(q-1) * sigma_vm_e / sigma_yield
    ds_drho = q * (rho ** (q - 1)) * (sigma_vm / sigma_yield)

    return dt_ds * ds_drho


def compute_stress_adjoint_rhs(
    sigma_vm:    np.ndarray,
    rho:         np.ndarray,
    u_full:      np.ndarray,
    mesh:        VoxelMesh,
    asm:         FEAssembler,
    material:    MaterialModel,
    sigma_yield: float,
    P:           int   = 6,
    q:           float = 0.5,
) -> np.ndarray:
    """
    Assemble the adjoint right-hand side g for the stress adjoint solve (D29).

    The adjoint equation is: K lambda = g
    where g = d(tilde_sigma)/d(u) = sum_e B_e^T d(tilde_sigma)/d(eps_e)

    Returns
    -------
    g : (n_dof,) adjoint right-hand side
    """
    h_vec = np.array([mesh.lx / mesh.nx,
                      mesh.ly / mesh.ny,
                      mesh.lz / mesh.nz])
    B_c, _ = asm._strain_displacement(0.0, 0.0, 0.0, h_vec)  # (6, 24)
    v_elem = (mesh.lx * mesh.ly * mesh.lz) / mesh.n_elem

    s_norm = (rho ** q) * (sigma_vm / sigma_yield)
    s_max  = s_norm.max()
    if s_max < 1e-30:
        return np.zeros(mesh.n_dof)

    s_rel  = s_norm / s_max
    p_sum  = np.sum(s_rel ** P)
    tilde  = s_max * (p_sum ** (1.0 / P))
    if tilde < 1e-30:
        return np.zeros(mesh.n_dof)

    # d(tilde)/d(sigma_vm_e) = tilde^(1-P) * s_norm_e^(P-1) * rho_e^q / sigma_yield
    dt_dsvm = (
        (tilde ** (1 - P)) *
        (s_norm ** (P - 1)) *
        (rho ** q) / sigma_yield
    )

    Ue  = u_full[mesh.dof_map]       # (n_elem, 24)
    eps = Ue @ B_c.T                 # (n_elem, 6)
    sig = eps @ material.C0.T        # (n_elem, 6)

    # Avoid division by zero in void elements
    vm_safe = np.where(sigma_vm > 1e-10, sigma_vm, 1e-10)

    # d(sigma_vm)/d(sig): (n_elem, 6)
    dsvm_dsig = np.zeros((mesh.n_elem, 6))
    dsvm_dsig[:, 0] = (sig[:,0] - sig[:,1] + sig[:,0] - sig[:,2]) / (2 * vm_safe)
    dsvm_dsig[:, 1] = (sig[:,1] - sig[:,0] + sig[:,1] - sig[:,2]) / (2 * vm_safe)
    dsvm_dsig[:, 2] = (sig[:,2] - sig[:,0] + sig[:,2] - sig[:,1]) / (2 * vm_safe)
    dsvm_dsig[:, 3] = 3.0 * sig[:, 3] / vm_safe
    dsvm_dsig[:, 4] = 3.0 * sig[:, 4] / vm_safe
    dsvm_dsig[:, 5] = 3.0 * sig[:, 5] / vm_safe

    # d(tilde)/d(sig_e): (n_elem, 6)
    dt_dsig = dt_dsvm[:, np.newaxis] * dsvm_dsig

    # d(tilde)/d(eps_e): (n_elem, 6)
    dt_deps = dt_dsig @ material.C0

    # Assemble adjoint RHS: g = sum_e B_e^T d(tilde)/d(eps_e) * v_elem
    g_elem = (dt_deps @ B_c) * v_elem      # (n_elem, 24)

    g = np.zeros(mesh.n_dof)
    np.add.at(g, mesh.dof_map.ravel(), g_elem.ravel())
    return g


def stress_adjoint_sensitivity(
    K_ff:        object,
    g:           np.ndarray,
    bc,
    rho:         np.ndarray,
    sigma_vm:    np.ndarray,
    mesh:        VoxelMesh,
    asm:         FEAssembler,
    material:    MaterialModel,
    sigma_yield: float,
    P:           int   = 6,
    q:           float = 0.5,
    p_simp:      float = 3.0,
    E_min:       float = 1e-9,
) -> np.ndarray:
    """
    Compute full adjoint stress sensitivity (D29).

    Solves K lambda = g[free], then assembles direct + adjoint terms.
    Note: full adjoint term is assembled inline in simp.py where u_full is available.

    Returns
    -------
    ds_drho : (n_elem,) direct sensitivity (adjoint term added in simp.py)
    """
    import scipy.sparse.linalg as spla

    g_f   = g[bc.free_dofs]
    lam_f = spla.spsolve(K_ff, -g_f)
    lam_full = bc.expand(lam_f)

    ds_direct = p_norm_sensitivity_direct(sigma_vm, rho, sigma_yield, P, q)
    return ds_direct


# ── Augmented Lagrangian penalty ──────────────────────────────────────────────

class StressConstraintHandler:
    """
    Manages the augmented Lagrangian stress constraint (D31).

    The augmented Lagrangian adds to the compliance objective:
        L(rho, mu, lam) = C(rho) + mu/2 * max(0, tilde_sigma - 1 + lam/mu)^2

    where:
        mu  : penalty weight (grows when constraint violated)
        lam : Lagrange multiplier estimate
        tilde_sigma : P-norm stress (constraint: <= 1)
    """

    def __init__(self, stress_cfg: StressConfig, sigma_yield: float):
        self.cfg          = stress_cfg
        self.sigma_yield  = sigma_yield
        self.sigma_limit  = sigma_yield / max(stress_cfg.safety_factor, 1e-6)
        self.mu           = stress_cfg.penalty_weight
        self.lam          = 0.0
        self.violation_history = []

    @property
    def active(self) -> bool:
        return self.cfg.safety_factor > 0

    def penalty(self, tilde_sigma: float) -> float:
        """
        Augmented Lagrangian penalty value.
        tilde_sigma is normalised by sigma_yield (constraint: <= 1).
        """
        if not self.active:
            return 0.0
        s_norm = tilde_sigma / (self.sigma_limit / self.sigma_yield)
        viol = max(0.0, s_norm - 1.0 + self.lam / self.mu)
        return 0.5 * self.mu * viol ** 2

    def sensitivity(self, tilde_sigma: float, ds_drho: np.ndarray) -> np.ndarray:
        """Sensitivity of the penalty term w.r.t. rho."""
        if not self.active:
            return np.zeros_like(ds_drho)
        s_norm = tilde_sigma / (self.sigma_limit / self.sigma_yield)
        viol = max(0.0, s_norm - 1.0 + self.lam / self.mu)
        if viol < 1e-10:
            return np.zeros_like(ds_drho)
        sigma_target = self.sigma_limit / self.sigma_yield
        return self.mu * viol * ds_drho / sigma_target

    def update(self, tilde_sigma: float):
        """Update Lagrange multiplier and penalty weight after each iteration."""
        if not self.active:
            return
        sigma_target = self.sigma_limit / self.sigma_yield
        s_norm = tilde_sigma / sigma_target
        viol   = s_norm - 1.0

        self.violation_history.append(float(viol))
        self.lam = max(0.0, self.lam + self.mu * viol)
        if viol > 0.01:
            self.mu = min(self.mu * self.cfg.penalty_update, 1e6)

    def report(self, tilde_sigma: float) -> dict:
        """Summary dict for the iteration log."""
        if not self.active:
            return {}
        sigma_target = self.sigma_limit / self.sigma_yield
        return {
            "tilde_sigma_norm": float(tilde_sigma / sigma_target),
            "violation":        float(tilde_sigma / sigma_target - 1.0),
            "mu":               float(self.mu),
            "lam":              float(self.lam),
        }

"""
simp.py — Chunk 2: SIMP topology optimisation inner loop

Implements the Solid Isotropic Material with Penalisation (SIMP) method
for compliance minimisation subject to a volume constraint.

Decisions implemented here:
    Decision 7  — Vectorised sensitivity via einsum (no Python loops)
    Decision 8  — Auto scheme: OC for single constraint, MMA for multi;
                  ML-guided switching in chunk 8 via diagnostics
    Decision 9  — Volume constraint by bisection, tol=1e-6, 100 iters max
    Decision 10 — SIMPResult NamedTuple with objectives + diagnostics

Entry point:
    result = run_simp(problem, design_params, config)

where problem = build_problem() from mesh_material.py.
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse.linalg as spla
from typing import NamedTuple
from dataclasses import dataclass, field

from gdto.mesh_material import ProblemData
from gdto.filters import DensityFilter, OverhangFilter


# ---------------------------------------------------------------------------
# SIMPResult — public return type consumed by GD outer loop (chunk 5)
# ---------------------------------------------------------------------------

class SIMPResult(NamedTuple):
    """
    Complete result of one SIMP inner loop evaluation.

    Primary objectives (fed to NSGA-II in chunk 5):
        compliance     — structural compliance C = f^T u  [N·m]
        mass           — part mass [kg]
        support_volume — smooth overhang proxy [m³]

    Density field (fed to chunk 4 geometry reconstruction):
        rho            — converged element densities, shape (n_elem,)

    Convergence diagnostics (fed to chunk 8 surrogate for scheme selection):
        converged           — did SIMP reach tol before max_iter?
        n_iterations        — number of SIMP iterations taken
        compliance_history  — compliance at each iteration
        change_history      — ||Δrho||_inf at each iteration
        bisection_iters_mean — mean bisection iters (for surrogate)
        mu_history          — Lagrange multiplier history (for surrogate)
        scheme_used         — 'OC' or 'MMA' (for surrogate training)
        wall_time_s         — total wall time [s]

    Design parameters that produced this result (surrogate training input):
        design_params  — dict with volume_fraction, overhang_weight, etc.
    """
    # Primary objectives
    compliance:     float
    mass:           float
    support_volume: float

    # Density field
    rho: np.ndarray

    # Convergence
    converged:          bool
    n_iterations:       int
    compliance_history: np.ndarray
    change_history:     np.ndarray

    # Bisection diagnostics
    bisection_iters_mean: float
    mu_history:           np.ndarray
    scheme_used:          str

    # Timing
    wall_time_s: float

    # Surrogate training
    design_params: dict


# ---------------------------------------------------------------------------
# SIMPConfig — runtime parameters
# ---------------------------------------------------------------------------

@dataclass
class SIMPConfig:
    """
    SIMP solver configuration.

    All parameters have defaults matching configs/default.yaml simp: block.
    Construct from dict via SIMPConfig.from_dict(cfg['simp']).
    """
    p:              float = 3.0      # penalisation exponent
    E_min_factor:   float = 1e-9     # void stiffness floor
    max_iter:       int   = 100      # maximum SIMP iterations
    tol:            float = 1e-3     # convergence: ||Δrho||_inf < tol
    volume_fraction: float = 0.4    # target volume fraction
    update_scheme:  str   = "auto"  # "auto" | "OC" | "MMA"
    rho_min:        float = 0.001   # minimum element density

    # OC parameters
    oc_move: float = 0.2   # maximum density move per iteration
    oc_eta:  float = 0.5   # OC damping exponent

    # Volume bisection parameters
    bisect_tol:      float = 1e-6   # volume constraint tolerance
    bisect_max_iter: int   = 100    # maximum bisection iterations
    bisect_safety:   float = 2.0    # upper bracket safety factor

    # Overhang proxy parameters
    overhang_threshold: float = 0.5  # density threshold for support proxy

    filter_radius: float = 1.5    # density filter radius in element lengths
    overhang_P:    float = 2.0    # overhang P-norm exponent
    build_axis:    int   = 2      # build direction: 0=X, 1=Y, 2=Z

    @classmethod
    def from_dict(cls, d: dict) -> "SIMPConfig":
        """Construct from a config dict (e.g. yaml['simp'])."""
        oc = d.get("oc", {})
        return cls(
            p               = float(d.get("p",               cls.p)),
            E_min_factor    = float(d.get("E_min_factor",    cls.E_min_factor)),
            max_iter        = int(d.get("max_iter",          cls.max_iter)),
            tol             = float(d.get("tol",             cls.tol)),
            volume_fraction = float(d.get("volume_fraction", cls.volume_fraction)),
            update_scheme   = str(d.get("update_scheme",     cls.update_scheme)),
            rho_min         = float(d.get("rho_min",         cls.rho_min)),
            oc_move         = float(oc.get("move",           cls.oc_move)),
            oc_eta          = float(oc.get("eta",            cls.oc_eta)),
            bisect_tol      = float(d.get("bisect_tol",      cls.bisect_tol)),
            bisect_max_iter = int(d.get("bisect_max_iter",   cls.bisect_max_iter)),
            bisect_safety   = float(d.get("bisect_safety",   cls.bisect_safety)),
            filter_radius   = float(d.get("filter_radius",   cls.filter_radius)),
            overhang_P      = float(d.get("overhang_P",      cls.overhang_P)),
            build_axis      = int(d.get("build_axis",         cls.build_axis)),
        )


# ---------------------------------------------------------------------------
# MMAState — cross-iteration state for MMA update scheme
# ---------------------------------------------------------------------------

@dataclass
class MMAState:
    """
    State carried across iterations for the MMA update scheme.
    Initialised on first MMA call, updated each iteration.

    Fields
    ------
    rho_km1 : np.ndarray  — rho at iteration k-1
    rho_km2 : np.ndarray  — rho at iteration k-2
    L       : np.ndarray  — lower asymptotes
    U       : np.ndarray  — upper asymptotes
    iter    : int         — current iteration count
    """
    rho_km1: np.ndarray
    rho_km2: np.ndarray
    L:       np.ndarray
    U:       np.ndarray
    iter:    int = 0


# ---------------------------------------------------------------------------
# SIMPSolver
# ---------------------------------------------------------------------------

class SIMPSolver:
    """
    SIMP inner loop solver.

    Usage
    -----
    solver = SIMPSolver(problem, config)
    result = solver.solve(design_params)

    Parameters
    ----------
    problem : ProblemData
        Fully constructed FE problem from build_problem().
    config : SIMPConfig
        Solver parameters.
    """

    def __init__(self, problem: ProblemData, config: SIMPConfig) -> None:
        self.problem = problem
        self.config  = config
        self._mma_state: MMAState | None = None  # ← existing line

        # Construct filters (Decision 11-13)
        self._density_filter  = DensityFilter(
            problem.mesh,
            radius=getattr(config, 'filter_radius', 1.5)
        )
        self._overhang_filter = OverhangFilter(
            problem.mesh,
            P          = getattr(config, 'overhang_P', 2.0),
            build_axis = getattr(config, 'build_axis', 2),
        )

    def solve(self, design_params: dict | None = None) -> SIMPResult:
        """
        Run the SIMP inner loop to convergence.

        Parameters
        ----------
        design_params : dict, optional
            Design variable values from the GD outer loop.
            Stored in SIMPResult for surrogate training.
            Keys: volume_fraction, overhang_weight, build_orientation_deg,
                  min_feature_size.

        Returns
        -------
        SIMPResult
        """
        if design_params is None:
            design_params = {}

        t_start = time.perf_counter()
        cfg     = self.config
        prob    = self.problem
        mesh    = prob.mesh
        mat     = prob.material
        asm     = prob.assembler
        bc      = prob.bc

        # Volume fraction — prefer design_params over config
        Vf = float(design_params.get("volume_fraction", cfg.volume_fraction))

        # Select update scheme
        scheme = self._select_scheme(design_params)

        # Initialise density field uniformly at Vf
        rho = np.full(mesh.n_elem, Vf, dtype=np.float64)

        # History buffers
        compliance_history = []
        change_history     = []
        mu_history         = []
        bisect_iters_list  = []

        converged = False

        # ── SIMP iteration loop ────────────────────────────────────────────
        for iteration in range(cfg.max_iter):

            # 1. Filter density field (Decision 11 forward pass)
            rho_filtered = self._density_filter.apply(rho)

            # 2. Assemble K and apply BCs using filtered density
            K         = asm.assemble_K(rho_filtered, p=cfg.p,
                                       E_min_factor=cfg.E_min_factor)
            K_ff, f_f = bc.apply(K)

            # 3. Solve reduced system
            u_free = self._solve_linear(K_ff, f_f)
            u_full = bc.expand(u_free)

            # 4. Compliance (Decision 10: reuse strain energies)
            Ue            = u_full[mesh.dof_map]
            Ke0_Ue        = Ue @ asm.Ke0.T
            strain_energy = np.einsum('ei,ei->e', Ue, Ke0_Ue)
            compliance    = float(np.dot(
                cfg.E_min_factor + rho_filtered**cfg.p * (1.0 - cfg.E_min_factor),
                strain_energy
            ))

            # 5. Sensitivity w.r.t filtered density (Decision 7)
            dc_d_rho_filtered = (
                -cfg.p
                * rho_filtered ** (cfg.p - 1.0)
                * (1.0 - cfg.E_min_factor)
                * strain_energy
            )

            # 6. Chain rule backprop through density filter (Decision 11)
            dc_drho_filtered = self._density_filter.backprop(dc_d_rho_filtered)

            # 7. Overhang constraint gradient (Decision 13)
            V_supp, dV_drho = self._overhang_filter.compute(rho_filtered)

            # 8. Density update
            rho_old = rho.copy()

            if scheme == "OC":
                rho, mu, bisect_iters = self._oc_update(
                    rho, dc_drho_filtered, Vf
                )
            else:  # MMA
                # extra_grads (overhang constraint) wired into dual
                # bisection in chunk 3 — currently volume only
                rho, mu, bisect_iters = self._mma_update(
                    rho, dc_drho_filtered, Vf, iteration
                )

            # 9. Compute change
            change = float(np.max(np.abs(rho - rho_old)))

            # 10. Log diagnostics
            compliance_history.append(compliance)
            change_history.append(change)
            mu_history.append(mu)
            bisect_iters_list.append(bisect_iters)

            # 11. Convergence check
            if change < cfg.tol:
                converged = True
                break

        # ── Post-loop: compute objectives ─────────────────────────────────

        # Mass (Decision 10)
        v_elem = (mesh.lx * mesh.ly * mesh.lz) / mesh.n_elem
        mass   = float(mat.rho * v_elem * rho.sum())

        # Support volume proxy (Decision 10: smooth directional filter)
        support_volume = self._compute_support_volume(rho, v_elem)

        wall_time = time.perf_counter() - t_start

        return SIMPResult(
            compliance          = compliance,
            mass                = mass,
            support_volume      = support_volume,
            rho                 = rho.copy(),
            converged           = converged,
            n_iterations        = len(compliance_history),
            compliance_history  = np.array(compliance_history),
            change_history      = np.array(change_history),
            bisection_iters_mean = float(np.mean(bisect_iters_list)),
            mu_history          = np.array(mu_history),
            scheme_used         = scheme,
            wall_time_s         = wall_time,
            design_params       = dict(design_params),
        )

    # ── Scheme selection (Decision 8) ─────────────────────────────────────

    def _select_scheme(self, design_params: dict) -> str:
        """
        Auto-select OC or MMA based on active constraints.

        Decision 8: 'auto' uses OC for single volume constraint,
        MMA when additional constraints are active.
        ML-guided selection implemented in chunk 8 via surrogate.
        """
        scheme = self.config.update_scheme
        if scheme != "auto":
            return scheme

        # Count active constraints beyond volume
        # Overhang is active if overhang_weight > 0
        extra_constraints = 0
        if float(design_params.get("overhang_weight", 0.0)) > 0:
            extra_constraints += 1

        return "OC" if extra_constraints == 0 else "MMA"

    # ── Linear solve (Decision 6) ─────────────────────────────────────────

    def _solve_linear(
        self, K_ff: object, f_f: np.ndarray
    ) -> np.ndarray:
        """
        Solve K_ff u_f = f_f using the configured solver.

        Decision 6: 'direct' uses spsolve (SuperLU),
        'cg' uses conjugate gradient with ILU preconditioner.
        """
        if self.problem.solver == "direct":
            return spla.spsolve(K_ff, f_f)
        else:
            M    = spla.spilu(K_ff)
            M_op = spla.LinearOperator(K_ff.shape, M.solve)
            u, info = spla.cg(K_ff, f_f, M=M_op, atol=1e-10)
            if info != 0:
                # Fall back to direct if CG fails
                return spla.spsolve(K_ff, f_f)
            return u

    # ── OC density update (Decision 8 + 9) ───────────────────────────────

    def _oc_update(
        self,
        rho:      np.ndarray,
        dc_drho:  np.ndarray,
        Vf:       float,
    ) -> tuple[np.ndarray, float, int]:
        """
        Optimality Criteria density update with bisection on volume constraint.

        Update rule (Sigmund 2001):
            B_e = (-dc_drho_e / mu)^eta
            rho_new_e = clip(rho_e * B_e, rho_e - move, rho_e + move)
                        clipped to [rho_min, 1]

        Volume constraint enforced by bisecting on Lagrange multiplier mu
        until mean(rho_new) = Vf to within bisect_tol.

        Parameters
        ----------
        rho      : current density field, shape (n_elem,)
        dc_drho  : filtered sensitivities, shape (n_elem,) — all negative
        Vf       : target volume fraction

        Returns
        -------
        rho_new      : updated density field
        mu_star      : converged Lagrange multiplier
        bisect_iters : number of bisection iterations taken
        """
        cfg  = self.config
        p    = cfg.oc_eta
        m    = cfg.oc_move
        rmin = cfg.rho_min

        # Sensitivities should be negative — use absolute values for bracket
        # s_e = -dc_drho_e > 0
        s = -dc_drho
        s = np.maximum(s, 1e-30)  # guard against numerical zero

        # Bisection bracket (Decision 9)
        # Lower: mu_lo -> B_e large -> all rho pushed to upper bound -> vol > Vf
        # Upper: mu_hi -> B_e small -> all rho pushed to lower bound -> vol < Vf
        mu_lo = 1e-10 * s.min()
        mu_hi = cfg.bisect_safety * s.max() / (1.0 - m) ** p

        def _update(mu: float) -> np.ndarray:
            Be      = (s / mu) ** p
            rho_new = rho * Be
            rho_new = np.clip(rho_new, rho - m, rho + m)
            rho_new = np.clip(rho_new, rmin, 1.0)
            return rho_new

        # Bisection loop
        bisect_iters = 0
        mu = 0.5 * (mu_lo + mu_hi)
        for bisect_iters in range(1, cfg.bisect_max_iter + 1):
            mu       = 0.5 * (mu_lo + mu_hi)
            rho_new  = _update(mu)
            vol_err  = rho_new.mean() - Vf

            if abs(vol_err) < cfg.bisect_tol:
                break

            if vol_err > 0:
                mu_lo = mu   # volume too high — increase mu to reduce densities
            else:
                mu_hi = mu   # volume too low  — decrease mu to increase densities

        return rho_new, float(mu), bisect_iters

    # ── MMA density update (Decision 8 + 9) ──────────────────────────────

    def _mma_update(
        self,
        rho:      np.ndarray,
        dc_drho:  np.ndarray,
        Vf:       float,
        iteration: int,
    ) -> tuple[np.ndarray, float, int]:
        """
        Method of Moving Asymptotes density update.

        Implements Svanberg (1987) MMA with:
            - Oscillation-adaptive asymptotes (Section 4 of Decision 8)
            - Closed-form per-element solution via dual bisection
            - Single volume constraint enforced by bisection on lambda_1

        State is carried in self._mma_state across iterations.
        On first call (iteration 0 or state is None), state is initialised.

        Parameters
        ----------
        rho       : current density field
        dc_drho   : filtered sensitivities (all negative for compliance)
        Vf        : target volume fraction
        iteration : current SIMP iteration index

        Returns
        -------
        rho_new      : updated density field
        lambda1_star : converged dual variable
        bisect_iters : bisection iterations taken
        """
        cfg  = self.config
        n    = len(rho)
        rmin = cfg.rho_min

        # Box constraints
        alpha = np.full(n, rmin)
        beta  = np.ones(n)
        rho_range = beta - alpha  # = 1 - rho_min for all elements

        # ── Asymptote update ───────────────────────────────────────────────
        s0 = 0.5  # initial asymptote fraction

        if self._mma_state is None or iteration == 0:
            # First iteration — initialise state, no oscillation history
            L = rho - s0 * rho_range
            U = rho + s0 * rho_range
            self._mma_state = MMAState(
                rho_km1 = rho.copy(),
                rho_km2 = rho.copy(),
                L       = L,
                U       = U,
                iter    = 0,
            )
        else:
            state   = self._mma_state
            rho_km1 = state.rho_km1
            rho_km2 = state.rho_km2
            L_km1   = state.L
            U_km1   = state.U

            if state.iter < 2:
                # Not enough history for oscillation detection
                gamma = np.zeros(n)
            else:
                gamma = (rho - rho_km1) * (rho_km1 - rho_km2)

            # Contraction/expansion factors
            s_factor = np.where(gamma < 0, 0.7, np.where(gamma > 0, 1.2, 1.0))

            L = rho - s_factor * (rho_km1 - L_km1)
            U = rho + s_factor * (U_km1 - rho_km1)

            # Clamp asymptotes to safe range
            L = np.maximum(L, rho - 10.0 * rho_range)
            L = np.minimum(L, rho - 0.01 * rho_range)
            U = np.minimum(U, rho + 10.0 * rho_range)
            U = np.maximum(U, rho + 0.01 * rho_range)

            self._mma_state = MMAState(
                rho_km1 = rho.copy(),
                rho_km2 = rho_km1.copy(),
                L       = L,
                U       = U,
                iter    = state.iter + 1,
            )

        # Tightened box constraints (Section 3 of Decision 8)
        alpha_k = np.maximum(alpha, L + 0.1 * (rho - L))
        beta_k  = np.minimum(beta,  U - 0.1 * (U - rho))

        # ── Compute p0e, q0e from objective sensitivity ────────────────────
        # (Section 3 of Decision 8 derivation)
        dfdp   = np.maximum(dc_drho,  0.0)   # (df/drho)^+
        dfdm   = np.maximum(-dc_drho, 0.0)   # (df/drho)^-
        p0e    = (U - rho) ** 2 * dfdp
        q0e    = (rho - L) ** 2 * dfdm

        # ── Compute p1e, q1e from volume constraint sensitivity ────────────
        # Volume constraint: g1 = mean(rho) - Vf
        # dg1/drho_e = 1/n for all elements (uniform mesh)
        dg1    = np.full(n, 1.0 / n)
        p1e    = (U - rho) ** 2 * dg1        # all positive since dg1 > 0
        q1e    = np.zeros(n)                  # dg1 >= 0 so q1e = 0

        # ── Dual bisection on lambda_1 ─────────────────────────────────────

        def _rho_star(lam1: float) -> np.ndarray:
            """Per-element closed-form MMA solution at given lambda_1."""
            Pe      = p0e + lam1 * p1e
            Qe      = q0e + lam1 * q1e + 1e-30  # guard sqrt(0)
            rho_new = (np.sqrt(Pe) * L + np.sqrt(Qe) * U) / (
                np.sqrt(Pe) + np.sqrt(Qe)
            )
            return np.clip(rho_new, alpha_k, beta_k)

        # Bracket for lambda_1
        lam_lo = 0.0
        # Bracket upper bound based on q0e (objective negative-gradient term).
        # p0e = 0 for compliance (all dc_drho negative) so cannot use it here.
        # Need lam1 large enough that sqrt(lam1*p1e) >> sqrt(q0e),
        # driving rho_star toward L (lower asymptote) → mean(rho) < Vf.
        lam_hi = cfg.bisect_safety * np.max(q0e + p0e + 1e-30) / np.min(p1e + 1e-30)

        bisect_iters = 0
        lam1 = 0.5 * (lam_lo + lam_hi)
        for bisect_iters in range(1, cfg.bisect_max_iter + 1):
            lam1    = 0.5 * (lam_lo + lam_hi)
            rho_new = _rho_star(lam1)
            vol_err = rho_new.mean() - Vf

            if abs(vol_err) < cfg.bisect_tol:
                break

            if vol_err > 0:
                lam_lo = lam1
            else:
                lam_hi = lam1

        return rho_new, float(lam1), bisect_iters

    # ── Support volume proxy (Decision 10) ───────────────────────────────

    def _compute_support_volume(
        self, rho: np.ndarray, v_elem: float
    ) -> float:
        V_supp, _ = self._overhang_filter.compute(rho)
        return V_supp
        """
        Smooth overhang support volume proxy (Gaynor & Guest 2016).

        For each element, the overhang measure is:
            o_e = max(0, rho_e - rho_below_e)

        where rho_below_e is the density of the element directly below e
        in the build direction (Z axis = axis 2 in the 3D grid).

        Support volume = v_elem * sum(o_e)

        This is smooth and differentiable in rho, enabling it to be used
        as a constraint sensitivity for MMA in chunk 3.
        """
        mesh     = self.problem.mesh
        rho_3d   = rho.reshape(mesh.nx, mesh.ny, mesh.nz)

        # Shift density field up by 1 in Z to get "below" values
        # rho_below[:,:,k] = rho_3d[:,:,k-1] (element below in Z)
        # Bottom layer (k=0) has no element below — treat as fully supported
        rho_below          = np.zeros_like(rho_3d)
        rho_below[:, :, 1:] = rho_3d[:, :, :-1]

        overhang       = np.maximum(0.0, rho_3d - rho_below)
        support_volume = float(v_elem * overhang.sum())
        return support_volume


# ---------------------------------------------------------------------------
# run_simp — convenience entry point
# ---------------------------------------------------------------------------

def run_simp(
    problem:       ProblemData,
    design_params: dict | None  = None,
    config:        SIMPConfig | None = None,
) -> SIMPResult:
    """
    Run the SIMP inner loop. Convenience wrapper around SIMPSolver.

    Parameters
    ----------
    problem : ProblemData
        From build_problem() in mesh_material.py.
    design_params : dict, optional
        GD outer loop design variables. Keys:
            volume_fraction      (overrides config if present)
            overhang_weight      (activates MMA if > 0)
            build_orientation_deg
            min_feature_size
    config : SIMPConfig, optional
        Solver parameters. Uses defaults if not provided.

    Returns
    -------
    SIMPResult
    """
    if config is None:
        config = SIMPConfig()
    solver = SIMPSolver(problem, config)
    return solver.solve(design_params)
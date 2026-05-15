"""
gd_loop.py — Chunk 5: NSGA-II outer loop for generative design

Wraps the SIMP inner loop as a pymoo multi-objective Problem and runs
NSGA-II to build a Pareto front over (compliance, mass, support_volume).

Design variables (4):
    x[0]  volume_fraction       [0.1, 0.9]
    x[1]  overhang_weight       [0.0, 1.0]   — activates MMA when > 0
    x[2]  build_orientation_deg [0.0, 90.0]  — STUB (chunk 7)
    x[3]  min_feature_size      [1.0, 4.0]   — maps to filter_radius

Objectives (3):
    f[0]  compliance_Nm    [N·m]  — structural flexibility (minimise)
    f[1]  mass_kg          [kg]   — material cost (minimise)
    f[2]  support_volume_m3 [m³]  — printability cost (minimise)

Decisions implemented:
    D23 — NSGA-II config: pop=20, gen=3 default, SBX η=15, PM η=20
    D24 — Objective interface: pymoo Problem, normalised by gen-1 reference
    D25 — Parallelisation: joblib loky backend (macOS-safe), n_jobs=cpu//2
    D26 — Pareto output: ParetoSolution dataclass, JSON front, STL on demand
"""

from __future__ import annotations

import os
import time
import logging
import traceback
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Callable

from joblib import Parallel, delayed

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from gdto.mesh_material import build_problem
from gdto.simp import run_simp, SIMPConfig, ThermalConfig
from gdto.objectives import compute_objectives, normalise_objectives


log = logging.getLogger(__name__)


# ── Design variable bounds ───────────────────────────────────────────────────

XL = np.array([0.1, 0.0,  0.0, 1.0])   # lower bounds
XU = np.array([0.9, 1.0, 90.0, 4.0])   # upper bounds

VAR_NAMES = [
    "volume_fraction",
    "overhang_weight",
    "build_orientation_deg",   # stub — chunk 7
    "min_feature_size",
]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class GDConfig:
    """
    Configuration for the generative design outer loop.

    pop_size    : NSGA-II population size (D23)
    n_gen       : maximum generations (D23)
    n_workers   : parallel SIMP evaluations (D25)
    simp_iter   : SIMP max_iter per evaluation (keep low for speed)
    simp_tol    : SIMP convergence tolerance
    mesh_nx     : voxel mesh resolution
    mesh_ny     :
    mesh_nz     :
    material    : material preset name
    fixed_faces : structural BCs
    load_face   : load face name
    load_dir    : load direction (0=X, 1=Y, 2=Z)
    load_mag    : load magnitude [N]
    lx, ly, lz : domain dimensions [m]
    """
    pop_size:    int   = 20
    n_gen:       int   = 3
    n_workers:   int   = 1      # unused directly; joblib uses cpu_count//2
    simp_iter:   int   = 15
    simp_tol:    float = 5e-2
    mesh_nx:     int   = 12
    mesh_ny:     int   = 10
    mesh_nz:     int   = 8
    material:    str   = "Ti64"
    fixed_faces: list  = field(default_factory=lambda: ["zmin"])
    load_face:   str   = "zmax"
    load_dir:    int   = 2
    load_mag:    float = -1000.0
    lx:          float = 1.0
    ly:          float = 1.0
    lz:          float = 1.0
    flux_faces:  dict  = field(default_factory=dict)
    temp_faces:  dict  = field(default_factory=dict)


@dataclass
class ParetoSolution:
    """One solution on the Pareto front."""
    solution_id:           int
    volume_fraction:       float
    overhang_weight:       float
    build_orientation_deg: float
    min_feature_size:      float
    compliance_Nm:         float
    mass_kg:               float
    support_volume_m3:     float
    rank:                  int   = 0
    crowding_distance:     float = 0.0
    stl_b64:               str | None = None
    report:                dict | None = None

    def to_dict(self) -> dict:
        cd = self.crowding_distance
        if cd is None or (isinstance(cd, float) and (cd != cd or cd == float('inf'))):
            cd = 0.0
        return {
            "solution_id":           int(self.solution_id),
            "volume_fraction":       float(self.volume_fraction),
            "overhang_weight":       float(self.overhang_weight),
            "build_orientation_deg": float(self.build_orientation_deg),
            "min_feature_size":      float(self.min_feature_size),
            "compliance_Nm":         float(self.compliance_Nm),
            "mass_kg":               float(self.mass_kg),
            "support_volume_m3":     float(self.support_volume_m3),
            "rank":                  int(self.rank),
            "crowding_distance":     float(cd),
            "report":                self.report,
        }
        # stl_b64 deliberately omitted — fetched on demand via /reconstruct_solution


@dataclass
class GDResult:
    """Complete result from the generative design loop."""
    pareto_solutions:  list[ParetoSolution]
    n_generations:     int
    n_evaluations:     int
    wall_time_s:       float
    objective_names:   list[str]
    objective_ref:     list[float]   # reference values for normalisation
    all_objectives:    np.ndarray    # (n_eval, 3) all evaluated objectives
    all_design_params: np.ndarray    # (n_eval, 4) all evaluated design params
    config:            GDConfig

    def to_dict(self) -> dict:
        return {
            "n_solutions":     int(len(self.pareto_solutions)),
            "n_generations":   int(self.n_generations),
            "n_evaluations":   int(self.n_evaluations),
            "wall_time_s":     float(round(self.wall_time_s, 1)),
            "objective_names": list(self.objective_names),
            "objective_ref":   [float(v) for v in self.objective_ref],
            "solutions":       [s.to_dict() for s in self.pareto_solutions],
        }


# ── Worker function (runs in separate process) ───────────────────────────────

def _evaluate_individual(args: tuple) -> tuple:
    """
    Evaluate one design parameter vector by running SIMP.

    Parameters
    ----------
    args : (idx, x_vec, gd_config_dict)
        idx            : population index (returned for ordering)
        x_vec          : (4,) design parameter vector
        gd_config_dict : GDConfig serialised as dict (for pickling)

    Returns
    -------
    (idx, compliance_Nm, mass_kg, support_volume_m3, error_msg)
    error_msg is empty string on success.
    """
    try:
        idx, x, cfg_dict = args
        cfg = GDConfig(**cfg_dict)

        # Unpack design variables
        vf    = float(x[0])
        ow    = float(x[1])
        # x[2] = build_orientation_deg — stub, ignored until chunk 7
        mfs   = float(x[3])   # min_feature_size -> filter_radius

        design_params = {
            "volume_fraction":       vf,
            "overhang_weight":       ow,
            "build_orientation_deg": float(x[2]),
            "min_feature_size":      mfs,
        }

        # Build FE problem
        problem = build_problem(
            material       = cfg.material,
            nx=cfg.mesh_nx, ny=cfg.mesh_ny, nz=cfg.mesh_nz,
            lx=cfg.lx, ly=cfg.ly, lz=cfg.lz,
            fixed_faces    = cfg.fixed_faces,
            load_face      = cfg.load_face,
            load_direction = cfg.load_dir,
            load_magnitude = cfg.load_mag,
        )

        # SIMP config
        simp_cfg = SIMPConfig(
            max_iter          = cfg.simp_iter,
            tol               = cfg.simp_tol,
            volume_fraction   = vf,
            filter_radius     = mfs,
            update_scheme     = "auto",
            p_start           = 2.0,
            p_end             = 3.0,
            snapshot_interval = 0,    # disable snapshots in GD — saves memory
        )

        # Thermal config
        thermal_cfg = ThermalConfig(
            flux_faces = cfg.flux_faces,
            temp_faces = cfg.temp_faces,
        )

        # Run SIMP
        result = run_simp(problem, config=simp_cfg,
                          thermal_cfg=thermal_cfg,
                          design_params=design_params)

        objs = compute_objectives(result)
        return (idx, objs[0], objs[1], objs[2], "")

    except Exception as e:
        idx = args[0] if args else -1
        return (idx, 1e10, 1e10, 1e10, traceback.format_exc())


# ── pymoo Problem class ───────────────────────────────────────────────────────

class TopOptProblem(Problem):
    """
    pymoo Problem wrapping the SIMP inner loop (D24).

    Each _evaluate() call receives a (pop_size, 4) matrix X.
    Evaluations run in parallel using ProcessPoolExecutor (D25).
    Objectives are normalised by first-generation reference values.
    """

    def __init__(
        self,
        gd_config:        GDConfig,
        progress_callback: Callable | None = None,
    ):
        super().__init__(
            n_var    = 4,
            n_obj    = 3,
            n_constr = 0,
            xl       = XL.copy(),
            xu       = XU.copy(),
        )
        self.gd_config         = gd_config
        self.progress_callback = progress_callback
        self._ref              = None     # set after first generation
        self._n_eval           = 0
        self._current_gen      = 0
        self._all_objs         = []
        self._all_x            = []
        # serialise config for pickling
        self._cfg_dict = {
            k: v for k, v in gd_config.__dict__.items()
        }

    def _evaluate(self, X, out, *args, **kwargs):
        n   = X.shape[0]
        cfg = self.gd_config
        F_raw     = np.full((n, 3), 1e10)
        n_jobs    = max(1, (os.cpu_count() or 4) // 2)
        args_list = [(i, X[i], self._cfg_dict) for i in range(n)]

        if self.progress_callback:
            self.progress_callback({
                "type":   "gen_start",
                "gen":    self._current_gen + 1,
                "n_gen":  cfg.n_gen,
                "n_eval": n,
                "n_jobs": n_jobs,
            })

        try:
            # generator_unordered yields each result the moment it finishes
            results_iter = Parallel(
                n_jobs    = n_jobs,
                backend   = 'loky',
                return_as = 'generator_unordered',
            )(delayed(_evaluate_individual)(a) for a in args_list)

            for idx, c, m, v, err in results_iter:
                if err:
                    log.warning(f"Eval failed (idx={idx}): {err[:100]}")
                else:
                    F_raw[idx] = [c, m, v]

                self._n_eval += 1
                if self.progress_callback:
                    self.progress_callback({
                        "type":       "evaluation",
                        "eval_id":    self._n_eval,
                        "gen":        self._current_gen + 1,
                        "n_gen":      cfg.n_gen,
                        "x":          X[idx].tolist(),
                        "objectives": F_raw[idx].tolist(),
                        "error":      bool(err),
                    })

        except Exception as e:
            log.warning(f"Parallel failed ({e}) — running sequentially")
            for i, a in enumerate(args_list):
                idx, c, m, v, err = _evaluate_individual(a)
                if not err:
                    F_raw[i] = [c, m, v]
                self._n_eval += 1
                if self.progress_callback:
                    self.progress_callback({
                        "type":       "evaluation",
                        "eval_id":    self._n_eval,
                        "gen":        self._current_gen + 1,
                        "n_gen":      cfg.n_gen,
                        "x":          X[i].tolist(),
                        "objectives": F_raw[i].tolist(),
                        "error":      bool(err),
                    })

        self._all_objs.extend(F_raw.tolist())
        self._all_x.extend(X.tolist())

        if self._ref is None:
            valid = F_raw[F_raw[:, 0] < 1e9]
            self._ref = valid.max(axis=0) if len(valid) > 0 else np.ones(3)
            log.info(f"Reference objectives set: {self._ref}")

        out["F"] = normalise_objectives(F_raw, self._ref)


# ── Main GD runner ────────────────────────────────────────────────────────────

def run_gd(
    gd_config:         GDConfig,
    progress_callback: Callable | None = None,
) -> GDResult:
    """
    Run the full NSGA-II generative design loop.

    Parameters
    ----------
    gd_config         : GDConfig
    progress_callback : optional callable(dict) called with SSE event dicts
                        during optimisation — used by the FastAPI SSE stream.

    Returns
    -------
    GDResult with Pareto front and full evaluation history.
    """
    t_start = time.perf_counter()

    cfg = gd_config

    problem = TopOptProblem(cfg, progress_callback)

    algorithm = NSGA2(
        pop_size  = cfg.pop_size,
        sampling  = FloatRandomSampling(),
        crossover = SBX(prob=0.9, eta=15, vtype=float, repair=None),
        mutation  = PM(prob=1.0/4, eta=20),
        eliminate_duplicates = True,
    )

    from pymoo.termination import get_termination
    termination = get_termination("n_gen", cfg.n_gen)

    log.info(f"Starting NSGA-II: pop={cfg.pop_size}, max_gen={cfg.n_gen}")

    # Generation callback
    def on_generation(algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
        problem._current_gen = gen
        # Extract current Pareto front
        ranks = pop.get("rank")
        pareto_mask = ranks == 0
        n_pareto = int(pareto_mask.sum()) if ranks is not None else 0
        if progress_callback:
            progress_callback({
                "type":        "generation",
                "gen":         gen,
                "n_gen":       cfg.n_gen,
                "n_eval":      problem._n_eval,
                "pareto_size": n_pareto,
            })
        log.info(f"Generation {gen}/{cfg.n_gen} — "
                 f"evals={problem._n_eval}, pareto={n_pareto}")

    res = minimize(
        problem,
        algorithm,
        termination,
        callback    = on_generation,
        verbose     = False,
        save_history = False,
    )

    # ── Extract Pareto front ─────────────────────────────────────────────────
    pareto_X = res.X    # (n_pareto, 4)
    pareto_F = res.F    # (n_pareto, 3) normalised

    # Denormalise objectives
    ref = problem._ref if problem._ref is not None else np.ones(3)
    pareto_F_raw = pareto_F * ref

    # Build ParetoSolution list
    pareto_solutions = []
    for i in range(len(pareto_X)):
        x = pareto_X[i]
        f = pareto_F_raw[i]

        try:
            crowding_arr = res.pop.get("crowding")
            cd = float(crowding_arr[i]) if crowding_arr is not None else 0.0
            if cd != cd or cd == float('inf'):   # guard nan/inf
                cd = 0.0
        except Exception:
            cd = 0.0

        pareto_solutions.append(ParetoSolution(
            solution_id           = i,
            volume_fraction       = float(x[0]),
            overhang_weight       = float(x[1]),
            build_orientation_deg = float(x[2]),
            min_feature_size      = float(x[3]),
            compliance_Nm         = float(f[0]),
            mass_kg               = float(f[1]),
            support_volume_m3     = float(f[2]),
            rank                  = 0,
            crowding_distance     = cd,
        ))

    # Sort by compliance (most common user preference)
    pareto_solutions.sort(key=lambda s: s.compliance_Nm)

    wall_time = time.perf_counter() - t_start

    if progress_callback:
        progress_callback({
            "type":      "gd_done",
            "n_pareto":  len(pareto_solutions),
            "n_eval":    problem._n_eval,
            "wall_time": round(wall_time, 1),
        })

    return GDResult(
        pareto_solutions  = pareto_solutions,
        n_generations     = res.algorithm.n_gen,
        n_evaluations     = problem._n_eval,
        wall_time_s       = wall_time,
        objective_names   = ["compliance_Nm", "mass_kg", "support_volume_m3"],
        objective_ref     = ref.tolist(),
        all_objectives    = np.array(problem._all_objs),
        all_design_params = np.array(problem._all_x),
        config            = cfg,
    )
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
    D23 — NSGA-II config: pop=40, gen=5 default, SBX η=15, PM η=20
    D24 — Objective interface: pymoo Problem, normalised by gen-1 reference
    D25 — Parallelisation: ProcessPoolExecutor, max_workers=cpu_count-1
    D26 — Pareto output: ParetoSolution dataclass, JSON front, STL on demand
"""

from __future__ import annotations

import os
import time
import json
import logging
import traceback
import numpy as np
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from gdto.mesh_material import build_problem, ProblemData
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
    pop_size:    int   = 40
    n_gen:       int   = 5
    n_workers:   int   = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    simp_iter:   int   = 40
    simp_tol:    float = 1e-2
    mesh_nx:     int   = 20
    mesh_ny:     int   = 20
    mesh_nz:     int   = 20
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
        d = asdict(self)
        d.pop('stl_b64', None)    # omit STL from JSON by default
        return d


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
            "n_solutions":     len(self.pareto_solutions),
            "n_generations":   self.n_generations,
            "n_evaluations":   self.n_evaluations,
            "wall_time_s":     round(self.wall_time_s, 1),
            "objective_names": self.objective_names,
            "objective_ref":   self.objective_ref,
            "solutions": [s.to_dict() for s in self.pareto_solutions],
        }


# ── Worker function (runs in separate process) ───────────────────────────────

def _evaluate_individual(args: tuple) -> tuple[float, float, float, str]:
    """
    Evaluate one design parameter vector by running SIMP.
    Runs in a worker process — must be picklable.

    Parameters
    ----------
    args : (x_vec, gd_config_dict)
        x_vec          : (4,) design parameter vector
        gd_config_dict : GDConfig serialised as dict (for pickling)

    Returns
    -------
    (compliance_Nm, mass_kg, support_volume_m3, error_msg)
    error_msg is empty string on success.
    """
    try:
        x, cfg_dict = args
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
            max_iter        = cfg.simp_iter,
            tol             = cfg.simp_tol,
            volume_fraction = vf,
            filter_radius   = mfs,
            update_scheme   = "auto",
            p_start         = 2.0,
            p_end           = 3.0,
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
        return (objs[0], objs[1], objs[2], "")

    except Exception as e:
        return (1e10, 1e10, 1e10, traceback.format_exc())


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
        self._all_objs         = []
        self._all_x            = []
        # serialise config for pickling
        self._cfg_dict = {
            k: v for k, v in gd_config.__dict__.items()
        }

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate a population matrix X of shape (n, 4).
        Fills out["F"] with normalised objectives of shape (n, 3).
        """
        n = X.shape[0]
        cfg = self.gd_config
        args_list = [(X[i], self._cfg_dict) for i in range(n)]

        F_raw = np.full((n, 3), 1e10)

        # Parallel evaluation (D25)
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            futures = {
                executor.submit(_evaluate_individual, a): i
                for i, a in enumerate(args_list)
            }
            for future in as_completed(futures):
                i   = futures[future]
                res = future.result()
                c, m, v, err = res
                if err:
                    log.warning(f"Eval {i} failed: {err[:200]}")
                else:
                    F_raw[i] = [c, m, v]
                self._n_eval += 1
                if self.progress_callback:
                    self.progress_callback({
                        "type":       "evaluation",
                        "eval_id":    self._n_eval,
                        "x":          X[i].tolist(),
                        "objectives": F_raw[i].tolist(),
                        "error":      bool(err),
                    })

        # Store all raw objectives and design params
        self._all_objs.extend(F_raw.tolist())
        self._all_x.extend(X.tolist())

        # Set reference values from first generation (D24)
        if self._ref is None:
            valid = F_raw[F_raw[:, 0] < 1e9]
            if len(valid) > 0:
                self._ref = valid.max(axis=0)
            else:
                self._ref = np.ones(3)
            log.info(f"Reference objectives set: {self._ref}")

        # Normalise
        F_norm = normalise_objectives(F_raw, self._ref)
        out["F"] = F_norm


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

    log.info(f"Starting NSGA-II: pop={cfg.pop_size}, max_gen={cfg.n_gen}, "
             f"workers={cfg.n_workers}")

    # Generation callback
    def on_generation(algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
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

        cd = float(res.pop.get("crowding")[i]) if hasattr(res.pop, 'get') else 0.0

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
"""
server.py — Chunk 4, Decision 16: FastAPI local server

Bridges the HTML frontend (Decision 15) with the Python TO pipeline.

Start with:
    uvicorn gdto.server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /optimise   — run SIMP + reconstruct + verify, streams SSE progress
    POST /verify     — run verification FEA only on uploaded STL
    GET  /materials  — return available material presets
    GET  /health     — confirm server is running
"""

from __future__ import annotations

import asyncio
import base64
import os
import json
import queue as _queue
import tempfile
import traceback
import threading
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gdto.mesh_material import build_problem, build_problem_dynamic, MaterialModel
from gdto.simp import run_simp, SIMPConfig, ThermalConfig
from gdto.reconstruct import reconstruct_stl, stl_to_base64, base64_to_stl, stl_to_voxels
from gdto.gd_loop import run_gd, GDConfig, GDResult
from gdto.verify import verify, THERMAL_CONSTANTS
from gdto.bc_projection import BCProjector, SurfacePatch, flood_fill_patch


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GD-TO Optimisation Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LoadConfig(BaseModel):
    face:       str   = "zmax"
    load_type:  str   = "force"       # force | pressure | flux | temperature
    value:      float = -1000.0
    direction:  int   = 2             # 0=X, 1=Y, 2=Z (for force only)


class PatchRequest(BaseModel):
    """
    A surface patch with BC — sent from the frontend after user selection.
    tri_indices : list of selected triangle indices into the uploaded STL
    bc_type     : 'force' | 'pressure' | 'fixed' | 'flux' | 'temperature'
    value       : scalar magnitude
    direction   : [x,y,z] unit vector (force/flux) or null
    label       : user label e.g. "bolt face"
    """
    tri_indices: list[int]
    bc_type:     str         = "force"
    value:       float       = 0.0
    direction:   list[float] | None = None
    label:       str         = ""


class FloodFillRequest(BaseModel):
    """Request to flood-fill a patch from a seed triangle."""
    stl_b64:       str
    seed_tri:      int
    angle_tol_deg: float = 25.0


class OptimiseRequest(BaseModel):
    stl_b64:         str              # base64-encoded STL
    material:        str   = "Ti64"
    nx:              int   = Field(20, ge=4, le=60)
    ny:              int   = Field(20, ge=4, le=60)
    nz:              int   = Field(20, ge=4, le=60)
    volume_fraction: float = Field(0.4, ge=0.1, le=0.9)
    fixed_faces:     list[str] = ["zmin"]
    loads:           list[LoadConfig] = []
    max_iter:        int   = Field(40, ge=5, le=200)
    filter_radius:   float = Field(1.5, ge=0.5, le=3.0)
    p_simp:          float = Field(3.0, ge=1.0, le=5.0)
    safety_factor:   float = Field(0.0, ge=0.0, le=10.0)  # 0 = disabled (D32)
    patches:         list[PatchRequest] | None = None


class VerifyRequest(BaseModel):
    stl_b64:       str
    material:      str   = "Ti64"
    nx:            int   = 20
    ny:            int   = 20
    nz:            int   = 20
    fixed_faces:   list[str] = ["zmin"]
    loads:         list[LoadConfig] = []


class GDRequest(BaseModel):
    """Request body for /gd_optimise endpoint."""
    stl_b64:         str
    material:        str   = "Ti64"
    nx:              int   = Field(20, ge=4, le=40)   # single-run mesh
    ny:              int   = Field(20, ge=4, le=40)
    nz:              int   = Field(20, ge=4, le=40)
    gd_nx:           int   = Field(12, ge=4, le=20)   # GD exploration mesh
    gd_ny:           int   = Field(10, ge=4, le=20)
    gd_nz:           int   = Field(8,  ge=4, le=20)
    pop_size:        int   = Field(20, ge=4, le=200)
    n_gen:           int   = Field(3,  ge=1, le=20)
    simp_iter:       int   = Field(15, ge=5, le=100)
    fixed_faces:     list[str] = ["zmin"]
    loads:           list[LoadConfig] = []


class ReconstructRequest(BaseModel):
    """Request to reconstruct STL for one Pareto solution."""
    stl_b64:         str        # original input STL (for voxelisation)
    design_params:   dict       # volume_fraction, overhang_weight, etc.
    material:        str  = "Ti64"
    nx:              int  = 20
    ny:              int  = 20
    nz:              int  = 20
    simp_iter:       int  = 80
    fixed_faces:     list[str] = ["zmin"]
    loads:           list[LoadConfig] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_loads(
    loads: list[LoadConfig],
) -> tuple[str, float, int, dict[str, float], dict[str, float], list[tuple]]:
    """
    Parse load list into BC parameters.

    Returns
    -------
    load_face, load_magnitude, load_direction,
    flux_faces, temp_faces,
    extra_forces — list of (face, magnitude, direction) for loads 2..N
    """
    force_loads = []    # all non-zero force/pressure loads in order
    flux_faces  = {}
    temp_faces  = {}

    for load in loads:
        if load.load_type in ("force", "pressure") and load.value != 0:
            force_loads.append((load.face, load.value, load.direction))
        elif load.load_type == "flux":
            flux_faces[load.face] = load.value
        elif load.load_type == "temperature":
            temp_faces[load.face] = load.value

    # Primary mechanical load — use negligible value when none specified so
    # the structural K remains well-conditioned (pure thermal problems are valid)
    if force_loads:
        load_face, load_magnitude, load_direction = force_loads[0]
    else:
        load_face, load_magnitude, load_direction = "zmax", -1e-6, 2

    extra_forces = force_loads[1:]   # loads 2..N applied via bc.add_load()
    return load_face, load_magnitude, load_direction, flux_faces, temp_faces, extra_forces


def _build_dynamic_bcs(
    patches:    list[PatchRequest],
    stl_verts:  np.ndarray,
    stl_faces:  np.ndarray,
    mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert frontend patch definitions to load vectors via BCProjector.
    Returns (f_structural, f_thermal, fixed_dofs, fixed_vals)
    """
    import logging

    projector  = BCProjector(mesh, stl_verts, stl_faces)
    patch_objs = []

    for p in patches:
        direction = np.array(p.direction) if p.direction else None
        patch_objs.append(SurfacePatch(
            tri_indices = np.array(p.tri_indices, dtype=np.int32),
            bc_type     = p.bc_type,
            value       = p.value,
            direction   = direction,
            label       = p.label,
        ))

    f_struct, f_therm, fixed_dofs, fixed_vals = projector.project_all(patch_objs)

    # ── Diagnostic ──
    logging.warning("BC projection summary:")
    logging.warning(f"  f_struct norm:      {np.linalg.norm(f_struct):.6e}")
    logging.warning(f"  f_struct max:       {np.abs(f_struct).max():.6e}")
    logging.warning(f"  fixed_dofs:         {len(fixed_dofs)} DOFs")
    logging.warning(f"  non-zero load DOFs: {np.sum(f_struct != 0)}")
    loaded_dofs = np.where(f_struct != 0)[0]
    overlap     = np.intersect1d(fixed_dofs, loaded_dofs)
    logging.warning(f"  overlap (fixed ∩ loaded): {len(overlap)} DOFs")
    if len(overlap) > 0:
        logging.warning(f"  WARNING: {len(overlap)} force DOFs are being zeroed by fixed BC")
    # ── End diagnostic ──

    return f_struct, f_therm, fixed_dofs, fixed_vals


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars/arrays to Python natives for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line (numpy-safe)."""
    return f"data: {json.dumps(data, cls=_NumpyEncoder)}\n\n"


# ---------------------------------------------------------------------------
# SSE generator for /optimise
# ---------------------------------------------------------------------------

async def _optimise_stream(req: OptimiseRequest) -> AsyncGenerator[str, None]:
    """
    Run the full pipeline and yield SSE events.

    Events:
        {type: "status",   message: str}
        {type: "progress", iteration: int, compliance: float, change: float}
        {type: "done",     stl_b64: str, report: dict, domain: dict}
        {type: "error",    message: str}
    """
    try:
        yield _sse({"type": "status", "message": "Decoding STL..."})
        await asyncio.sleep(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_stl  = Path(tmpdir) / "input.stl"
            out_stl = Path(tmpdir) / "optimised.stl"
            base64_to_stl(req.stl_b64, in_stl)

            # ── Voxelise STL into design domain ─────────────────────
            yield _sse({"type": "status", "message": "Voxelising design domain..."})
            await asyncio.sleep(0)

            rho_init, domain_info = stl_to_voxels(
                in_stl,
                nx=req.nx, ny=req.ny, nz=req.nz,
                volume_fraction=req.volume_fraction,
            )

            lx = domain_info["lx_m"]
            ly = domain_info["ly_m"]
            lz = domain_info["lz_m"]

            yield _sse({"type": "domain", "info": domain_info})
            await asyncio.sleep(0)

            # ── Build FE problem ─────────────────────────────────────
            yield _sse({"type": "status", "message": "Building FE problem..."})
            await asyncio.sleep(0)

            if req.patches:
                # Dynamic surface patches — arbitrary geometry BCs
                yield _sse({"type": "status",
                            "message": f"Projecting {len(req.patches)} BC patches..."})
                import trimesh as _trimesh
                stl_mesh_obj  = _trimesh.load(str(in_stl), force="mesh")
                stl_verts_arr = np.array(stl_mesh_obj.vertices, dtype=np.float64)
                stl_faces_arr = np.array(stl_mesh_obj.faces,    dtype=np.int32)

                from gdto.mesh_material import VoxelMesh
                mesh_obj = VoxelMesh(
                    nx=req.nx, ny=req.ny, nz=req.nz,
                    lx=lx, ly=ly, lz=lz,
                )

                f_struct, f_therm, fixed_dofs, fixed_vals = _build_dynamic_bcs(
                    req.patches, stl_verts_arr, stl_faces_arr, mesh_obj
                )

                from gdto.mesh_material import build_problem_dynamic
                problem = build_problem_dynamic(
                    material   = req.material,
                    nx=req.nx, ny=req.ny, nz=req.nz,
                    lx=lx, ly=ly, lz=lz,
                    f_load     = f_struct,
                    fixed_dofs = fixed_dofs,
                )

                flux_faces = {}
                temp_faces = {}
                thermal_cfg = ThermalConfig()

            else:
                # Legacy planar face BCs
                (load_face, load_magnitude, load_direction,
                 flux_faces, temp_faces, extra_forces) = _parse_loads(req.loads)

                problem = build_problem(
                    material       = req.material,
                    nx=req.nx, ny=req.ny, nz=req.nz,
                    lx=lx, ly=ly, lz=lz,
                    fixed_faces    = req.fixed_faces,
                    load_face      = load_face,
                    load_direction = load_direction,
                    load_magnitude = load_magnitude,
                )

                # Apply secondary loads directly to the BC load vector
                for ef_face, ef_mag, ef_dir in extra_forces:
                    problem.bc.add_load(ef_face, ef_dir, ef_mag)

                thermal_cfg = ThermalConfig(
                    flux_faces = flux_faces,
                    temp_faces = temp_faces,
                )

            # ── Run SIMP ─────────────────────────────────────────────
            yield _sse({"type": "status", "message": "Running topology optimisation..."})
            await asyncio.sleep(0)

            cfg = SIMPConfig(
                p               = req.p_simp,
                p_start         = 2.0,
                p_end           = req.p_simp,
                max_iter        = req.max_iter,
                tol             = 1e-3,
                volume_fraction = req.volume_fraction,
                filter_radius   = req.filter_radius,
                update_scheme   = "auto",
            )

            # Run SIMP in a thread, streaming progress events live via a queue
            progress_q: _queue.Queue = _queue.Queue()
            result_holder: list = [None]
            exc_holder:    list = [None]

            def _on_progress(iteration, compliance, change):
                progress_q.put((iteration, compliance, change))

            # Stress constraint config (D32)
            from gdto.stress_constraint import StressConfig as _StressConfig
            sf_val     = float(getattr(req, 'safety_factor', 0.0) or 0.0)
            stress_cfg = _StressConfig(
                safety_factor  = sf_val,
                p_norm         = 6,
                q_stress       = 0.5,
                penalty_weight = 1.0,
                penalty_update = 1.5,
            ) if sf_val > 0 else None

            def _run_simp():
                try:
                    result_holder[0] = run_simp(
                        problem, config=cfg,
                        thermal_cfg=thermal_cfg,
                        stress_cfg=stress_cfg,
                        on_progress=_on_progress,
                    )
                except Exception as e:
                    exc_holder[0] = e
                finally:
                    progress_q.put(None)  # sentinel — thread finished

            thread = threading.Thread(target=_run_simp, daemon=True)
            thread.start()

            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, progress_q.get)
                if item is None:
                    break
                iteration, compliance, change = item
                yield _sse({
                    "type":       "progress",
                    "iteration":  iteration,
                    "compliance": round(float(compliance), 6),
                    "change":     round(float(change), 6),
                })

            thread.join()
            if exc_holder[0] is not None:
                raise exc_holder[0]
            result = result_holder[0]

            import logging
            logging.warning(f"Compliance history: {result.compliance_history[:5]}")
            logging.warning(f"Compliance final: {result.compliance}")
            logging.warning(f"n_iterations: {result.n_iterations}")
            logging.warning(f"converged: {result.converged}")

            # ── Geometry reconstruction ───────────────────────────────
            yield _sse({"type": "status", "message": "Reconstructing geometry..."})
            await asyncio.sleep(0)

            stl_info = reconstruct_stl(
                result.rho, problem.mesh, out_stl,
                threshold=0.5, smooth_passes=2,
            )

            # ── Verification FEA ──────────────────────────────────────
            yield _sse({"type": "status", "message": "Running verification FEA..."})
            await asyncio.sleep(0)

            verify_result = await loop.run_in_executor(
                None, lambda: verify(
                    result, problem,
                    stl_path             = out_stl,
                    flux_faces           = flux_faces,
                    temp_faces           = temp_faces,
                    T_field_precomputed  = getattr(result, 'T_field_final', None),
                )
            )

            # ── Return results ────────────────────────────────────────
            stl_b64 = stl_to_base64(out_stl)

            # Encode density snapshots for iteration scrubber
            encoded_snaps = []
            encoded_iters = []
            if result.density_snapshots:
                for snap in result.density_snapshots:
                    encoded_snaps.append(
                        base64.b64encode(snap.tobytes()).decode()
                    )
                encoded_iters = result.snapshot_iters or []

            # Stress and temperature fields from verification FEA
            stress_b64 = None
            temp_b64   = None
            if verify_result.stress_field_mpa is not None:
                stress_b64 = base64.b64encode(
                    verify_result.stress_field_mpa.astype(np.float32).tobytes()
                ).decode()
            if verify_result.temp_field_c is not None:
                temp_b64 = base64.b64encode(
                    verify_result.temp_field_c.astype(np.float32).tobytes()
                ).decode()

            yield _sse({
                "type":       "done",
                "stl_b64":    stl_b64,
                "report":     verify_result.report,
                "simp": {
                    "scheme_used":   result.scheme_used,
                    "n_iterations":  result.n_iterations,
                    "converged":     result.converged,
                    "wall_time_s":   round(result.wall_time_s, 2),
                },
                "stl_info":   stl_info,
                "snapshots":  encoded_snaps,
                "snap_iters": encoded_iters,
                "stress_b64": stress_b64,
                "temp_b64":   temp_b64,
                "domain": {
                    "nx": req.nx, "ny": req.ny,
                      "nz": req.nz,
                    "lx_m": lx, "ly_m": ly, "lz_m": lz,
                },
            })

    except Exception as e:
        yield _sse({"type": "error", "message": str(e), "trace": traceback.format_exc()})


# ---------------------------------------------------------------------------
# SSE generator for /gd_optimise
# ---------------------------------------------------------------------------

async def _gd_stream(req: GDRequest) -> AsyncGenerator[str, None]:
    """
    Run full NSGA-II GD loop and stream SSE events.

    Events:
        {type: "status",      message: str}
        {type: "domain",      info: dict}
        {type: "gd_progress", n_gen: int, n_eval: int, n_pareto: int, wall_time: float}
        {type: "done",        pareto: dict}
        {type: "error",       message: str, trace: str}
    """
    try:
        yield _sse({"type": "status", "message": "Decoding STL..."})
        await asyncio.sleep(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_stl = Path(tmpdir) / "input.stl"
            base64_to_stl(req.stl_b64, in_stl)

            yield _sse({"type": "status", "message": "Voxelising design domain..."})
            await asyncio.sleep(0)

            rho_init, domain_info = stl_to_voxels(
                in_stl, nx=req.nx, ny=req.ny, nz=req.nz
            )
            lx = domain_info["lx_m"]
            ly = domain_info["ly_m"]
            lz = domain_info["lz_m"]

            yield _sse({"type": "domain", "info": domain_info})
            await asyncio.sleep(0)

            (load_face, load_magnitude, load_direction,
             flux_faces, temp_faces, _extra) = _parse_loads(req.loads)

            gd_cfg = GDConfig(
                pop_size    = req.pop_size,
                n_gen       = req.n_gen,
                simp_iter   = req.simp_iter,
                simp_tol    = 5e-2,
                mesh_nx     = req.gd_nx,
                mesh_ny     = req.gd_ny,
                mesh_nz     = req.gd_nz,
                material    = req.material,
                fixed_faces = req.fixed_faces,
                load_face   = load_face,
                load_dir    = load_direction,
                load_mag    = load_magnitude,
                lx=lx, ly=ly, lz=lz,
                flux_faces  = flux_faces,
                temp_faces  = temp_faces,
            )

            n_jobs = max(1, (os.cpu_count() or 4) // 2)
            yield _sse({
                "type":    "status",
                "message": f"Starting NSGA-II: pop={req.pop_size}, gen={req.n_gen}, "
                           f"est. {req.pop_size * req.n_gen} evaluations, "
                           f"{n_jobs} parallel workers",
            })
            await asyncio.sleep(0)

            yield _sse({"type": "status", "message": "Running NSGA-II..."})
            await asyncio.sleep(0)

            import queue as _queue
            import threading as _threading
            import time as _time

            event_q          = _queue.Queue()
            stop_evt         = _threading.Event()
            result_container = [None]
            error_container  = [None]

            def callback(ev: dict):
                event_q.put(ev)

            def heartbeat():
                t_start = _time.time()
                while not stop_evt.wait(timeout=3):
                    event_q.put({
                        "type":    "heartbeat",
                        "elapsed": round(_time.time() - t_start, 1),
                    })

            def run_in_thread():
                try:
                    result_container[0] = run_gd(gd_cfg, progress_callback=callback)
                except Exception as e:
                    error_container[0] = e
                finally:
                    stop_evt.set()
                    event_q.put({"type": "_done_sentinel"})

            _threading.Thread(target=run_in_thread, daemon=True).start()
            _threading.Thread(target=heartbeat,      daemon=True).start()

            while True:
                try:
                    ev = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: event_q.get(timeout=1.0)
                    )
                except Exception:
                    await asyncio.sleep(0)
                    continue

                if ev.get("type") == "_done_sentinel":
                    break

                yield _sse(ev)
                yield ""          # flush ASGI send buffer immediately
                await asyncio.sleep(0)

            stop_evt.set()

            if error_container[0]:
                raise error_container[0]

            gd_result: GDResult = result_container[0]

            yield _sse({
                "type":      "gd_progress",
                "n_gen":     gd_result.n_generations,
                "n_eval":    gd_result.n_evaluations,
                "n_pareto":  len(gd_result.pareto_solutions),
                "wall_time": round(gd_result.wall_time_s, 1),
            })

            yield _sse({"type": "status", "message": "Pareto front complete. Sending results..."})
            await asyncio.sleep(0)

            result_dict = gd_result.to_dict()

            import logging as _logging
            _logging.getLogger(__name__).info(
                f"GD done: {len(gd_result.pareto_solutions)} solutions, "
                f"{gd_result.n_evaluations} evals, {gd_result.wall_time_s:.1f}s"
            )

            yield _sse({"type": "done", "pareto": result_dict})
            yield ""               # flush ASGI send buffer
            await asyncio.sleep(0.2)

    except Exception as e:
        yield _sse({"type": "error", "message": str(e), "trace": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def shutdown_event():
    pass  # joblib loky workers are daemon processes — exit automatically


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/materials")
async def materials():
    presets = {}
    for name in ["Ti64", "AlSi10Mg", "316L"]:
        m = MaterialModel.from_preset(name)
        tc = THERMAL_CONSTANTS[name]
        presets[name] = {
            "E_perp_GPa":  round(m.E_perp / 1e9, 1),
            "E_par_GPa":   round(m.E_par  / 1e9, 1),
            "rho_kg_m3":   m.rho,
            "k_W_mK":      tc["k"],
            "alpha_per_K": tc["alpha"],
        }
    return presets


@app.post("/optimise")
async def optimise(req: OptimiseRequest):
    return StreamingResponse(
        _optimise_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/flood_fill")
async def flood_fill_endpoint(req: FloodFillRequest):
    """
    Server-side flood fill: given a seed triangle index,
    return the indices of all triangles in the same surface patch.
    Used by frontend after user clicks a triangle.
    """
    import trimesh as _trimesh

    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = Path(tmpdir) / "input.stl"
        base64_to_stl(req.stl_b64, stl_path)
        mesh_obj  = _trimesh.load(str(stl_path), force="mesh")
        verts     = np.array(mesh_obj.vertices, dtype=np.float64)
        faces     = np.array(mesh_obj.faces,    dtype=np.int32)

    selected = flood_fill_patch(
        req.seed_tri, verts, faces,
        angle_tol_deg=req.angle_tol_deg,
    )

    # Compute patch area and average normal for UI display
    v     = verts[faces[selected]]
    e1    = v[:, 1] - v[:, 0]
    e2    = v[:, 2] - v[:, 0]
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    norms = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)

    return {
        "tri_indices":    selected.tolist(),
        "n_triangles":    len(selected),
        "total_area_mm2": float(areas.sum()),
        "avg_normal":     norms.mean(axis=0).tolist(),
    }


@app.post("/verify")
async def verify_endpoint(req: VerifyRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        in_stl = Path(tmpdir) / "input.stl"
        base64_to_stl(req.stl_b64, in_stl)

        (load_face, load_magnitude, load_direction,
         flux_faces, temp_faces, extra_forces) = _parse_loads(req.loads)

        rho_init, domain_info = stl_to_voxels(
            in_stl,
            nx=req.nx, ny=req.ny, nz=req.nz,
        )

        problem = build_problem(
            material       = req.material,
            nx=req.nx, ny=req.ny, nz=req.nz,
            lx=domain_info["lx_m"],
            ly=domain_info["ly_m"],
            lz=domain_info["lz_m"],
            fixed_faces    = req.fixed_faces,
            load_face      = load_face,
            load_direction = load_direction,
            load_magnitude = load_magnitude,
        )
        for ef_face, ef_mag, ef_dir in extra_forces:
            problem.bc.add_load(ef_face, ef_dir, ef_mag)

        from gdto.simp import SIMPResult
        import time
        dummy = SIMPResult(
            compliance=1.0, mass=1.0, support_volume=0.0,
            rho=rho_init, converged=True, n_iterations=0,
            compliance_history=np.array([]), change_history=np.array([]),
            bisection_iters_mean=0.0, mu_history=np.array([]),
            scheme_used="OC", wall_time_s=0.0, design_params={},
        )
        result = verify(dummy, problem, flux_faces=flux_faces, temp_faces=temp_faces)
        return result.report


@app.post("/gd_optimise")
async def gd_optimise(req: GDRequest):
    """Run full NSGA-II generative design loop with SSE streaming."""
    return StreamingResponse(
        _gd_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/reconstruct_solution")
async def reconstruct_solution(req: ReconstructRequest):
    """
    Reconstruct STL for one Pareto solution with live SIMP progress.
    Streams SSE events during the solve then emits a final 'done' event.
    """
    return StreamingResponse(
        _reconstruct_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


async def _reconstruct_stream(req: ReconstructRequest):
    try:
        yield _sse({"type": "status", "message": "Preparing solution..."})
        await asyncio.sleep(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_stl  = Path(tmpdir) / "input.stl"
            out_stl = Path(tmpdir) / "solution.stl"
            base64_to_stl(req.stl_b64, in_stl)

            rho_init, domain_info = stl_to_voxels(
                in_stl, nx=req.nx, ny=req.ny, nz=req.nz,
                volume_fraction=req.design_params.get("volume_fraction", 0.4),
            )
            lx = domain_info["lx_m"]
            ly = domain_info["ly_m"]
            lz = domain_info["lz_m"]

            yield _sse({"type": "status",
                        "message": f"Building FE problem ({req.nx}×{req.ny}×{req.nz} mesh)..."})
            await asyncio.sleep(0)

            (load_face, load_magnitude, load_direction,
             flux_faces, temp_faces, _extra) = _parse_loads(req.loads)

            problem = build_problem(
                material       = req.material,
                nx=req.nx, ny=req.ny, nz=req.nz,
                lx=lx, ly=ly, lz=lz,
                fixed_faces    = req.fixed_faces,
                load_face      = load_face,
                load_direction = load_direction,
                load_magnitude = load_magnitude,
            )

            max_iter = req.simp_iter if req.simp_iter else 80
            cfg = SIMPConfig(
                max_iter        = max_iter,
                tol             = 1e-3,
                volume_fraction = req.design_params.get("volume_fraction", 0.4),
                filter_radius   = req.design_params.get("min_feature_size", 1.5),
                update_scheme   = "auto",
                p_start         = 2.0,
                p_end           = 3.0,
            )
            thermal_cfg = ThermalConfig(flux_faces=flux_faces, temp_faces=temp_faces)

            yield _sse({"type": "status", "message": f"Running SIMP ({max_iter} iterations)..."})
            await asyncio.sleep(0)

            event_q    = _queue.Queue()
            result_box = [None]
            error_box  = [None]

            def _simp_callback(iteration, compliance, change):
                event_q.put({
                    "type":       "progress",
                    "iteration":  int(iteration),
                    "max_iter":   max_iter,
                    "compliance": float(compliance),
                    "change":     float(change),
                })

            def _run_simp_thread():
                try:
                    result_box[0] = run_simp(
                        problem,
                        config            = cfg,
                        thermal_cfg       = thermal_cfg,
                        design_params     = req.design_params,
                        progress_callback = _simp_callback,
                    )
                except Exception as e:
                    error_box[0] = e
                finally:
                    event_q.put({"type": "_sentinel"})

            threading.Thread(target=_run_simp_thread, daemon=True).start()

            while True:
                try:
                    ev = event_q.get(timeout=1.0)
                except _queue.Empty:
                    await asyncio.sleep(0)
                    continue

                if ev.get("type") == "_sentinel":
                    break

                yield _sse(ev)
                await asyncio.sleep(0)

            if error_box[0]:
                raise error_box[0]

            result = result_box[0]

            yield _sse({"type": "status", "message": "Reconstructing geometry..."})
            await asyncio.sleep(0)

            stl_info = reconstruct_stl(result.rho, problem.mesh, out_stl)
            stl_b64  = stl_to_base64(out_stl)

            # ── Verification FEA (stress + thermal fields) ────────────
            yield _sse({"type": "status", "message": "Running verification FEA..."})
            await asyncio.sleep(0)

            loop = asyncio.get_event_loop()
            verify_result = await loop.run_in_executor(
                None, lambda: verify(
                    result, problem,
                    stl_path            = out_stl,
                    flux_faces          = flux_faces,
                    temp_faces          = temp_faces,
                    T_field_precomputed = getattr(result, 'T_field_final', None),
                )
            )

            # ── Encode density snapshots ──────────────────────────────
            encoded_snaps = []
            encoded_iters = []
            if result.density_snapshots:
                for snap in result.density_snapshots:
                    encoded_snaps.append(
                        base64.b64encode(snap.tobytes()).decode()
                    )
                encoded_iters = result.snapshot_iters or []

            # ── Encode stress / thermal fields ────────────────────────
            stress_b64 = None
            temp_b64   = None
            if verify_result.stress_field_mpa is not None:
                stress_b64 = base64.b64encode(
                    verify_result.stress_field_mpa.astype(np.float32).tobytes()
                ).decode()
            if verify_result.temp_field_c is not None:
                temp_b64 = base64.b64encode(
                    verify_result.temp_field_c.astype(np.float32).tobytes()
                ).decode()

            sf = None
            if verify_result and verify_result.min_safety_factor:
                sf_val = verify_result.min_safety_factor
                sf = float(sf_val) if sf_val != float('inf') else None

            yield _sse({
                "type":              "done",
                "stl_b64":           stl_b64,
                "stl_info":          stl_info,
                "stress_b64":        stress_b64,
                "temp_b64":          temp_b64,
                "snapshots":         encoded_snaps,
                "snap_iters":        encoded_iters,
                "compliance_Nm":     float(result.compliance),
                "mass_kg":           float(result.mass),
                "support_volume_m3": float(result.support_volume),
                "safety_factor":     sf,
                "domain": {
                    "nx": req.nx, "ny": req.ny, "nz": req.nz,
                    "lx_m": lx, "ly_m": ly, "lz_m": lz,
                },
            })
            await asyncio.sleep(0.1)

    except Exception as e:
        yield _sse({"type": "error", "message": str(e),
                    "trace": traceback.format_exc()})
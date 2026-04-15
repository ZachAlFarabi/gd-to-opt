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

from gdto.mesh_material import build_problem, MaterialModel
from gdto.simp import run_simp, SIMPConfig
from gdto.reconstruct import reconstruct_stl, stl_to_base64, base64_to_stl, stl_to_voxels
from gdto.verify import verify, THERMAL_CONSTANTS


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


class VerifyRequest(BaseModel):
    stl_b64:       str
    material:      str   = "Ti64"
    nx:            int   = 20
    ny:            int   = 20
    nz:            int   = 20
    fixed_faces:   list[str] = ["zmin"]
    loads:         list[LoadConfig] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_loads(
    loads: list[LoadConfig],
) -> tuple[int, float, int, dict[str, float], dict[str, float]]:
    """
    Parse load list into BC parameters.

    Returns
    -------
    load_face, load_magnitude, load_direction, flux_faces, temp_faces
    """
    load_face      = "zmax"
    load_magnitude = -1000.0
    load_direction = 2
    flux_faces     = {}
    temp_faces     = {}

    for load in loads:
        if load.load_type == "force":
            load_face      = load.face
            load_magnitude = load.value
            load_direction = load.direction
        elif load.load_type == "pressure":
            load_face      = load.face
            load_magnitude = load.value
            load_direction = load.direction
        elif load.load_type == "flux":
            flux_faces[load.face] = load.value
        elif load.load_type == "temperature":
            temp_faces[load.face] = load.value

    return load_face, load_magnitude, load_direction, flux_faces, temp_faces


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


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

            (load_face, load_magnitude, load_direction,
             flux_faces, temp_faces) = _parse_loads(req.loads)

            problem = build_problem(
                material       = req.material,
                nx=req.nx, ny=req.ny, nz=req.nz,
                lx=lx, ly=ly, lz=lz,
                fixed_faces    = req.fixed_faces,
                load_face      = load_face,
                load_direction = load_direction,
                load_magnitude = load_magnitude,
            )

            # ── Run SIMP ─────────────────────────────────────────────
            yield _sse({"type": "status", "message": "Running topology optimisation..."})
            await asyncio.sleep(0)

            cfg = SIMPConfig(
                p               = req.p_simp,
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

            def _run_simp():
                try:
                    result_holder[0] = run_simp(
                        problem, config=cfg, on_progress=_on_progress
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
                    stl_path   = out_stl,
                    flux_faces = flux_faces,
                    temp_faces = temp_faces,
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
                "domain": {
                    "nx": req.nx, "ny": req.ny, "nz": req.nz,
                    "lx_m": lx, "ly_m": ly, "lz_m": lz,
                },
            })

    except Exception as e:
        yield _sse({"type": "error", "message": str(e), "trace": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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


@app.post("/verify")
async def verify_endpoint(req: VerifyRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        in_stl = Path(tmpdir) / "input.stl"
        base64_to_stl(req.stl_b64, in_stl)

        (load_face, load_magnitude, load_direction,
         flux_faces, temp_faces) = _parse_loads(req.loads)

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
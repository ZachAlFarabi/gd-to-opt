"""
reconstruct.py — Chunk 4, Decision 14: geometry reconstruction

Converts a converged SIMP density field into a printable STL via:
    1. Threshold filtered density at tau (default 0.5)
    2. Marching cubes isosurface extraction (skimage)
    3. Taubin smoothing to remove voxel staircase artifacts (trimesh)
    4. STL export in physical metres

Reference: Lorensen & Cline (1987), Marching Cubes.
           Taubin (1995), Signal Processing Approach to Mesh Smoothing.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from gdto.mesh_material import VoxelMesh


def reconstruct_stl(
    rho, mesh, output_path,
    threshold=0.5,
    smooth_passes=10,    # was 2
    lamb=0.5,
    nu=0.53,
):
    """
    Reconstruct geometry from converged density field and export as STL.

    Pipeline (Decision 14):
        1. Reshape rho to 3D grid
        2. Marching cubes at isovalue = threshold
        3. Taubin smoothing (lamb/nu, smooth_passes iterations)
        4. Export STL to output_path

    Parameters
    ----------
    rho : np.ndarray, shape (n_elem,)
        Converged filtered density field from SIMPResult.
    mesh : VoxelMesh
        The mesh the density was computed on.
    output_path : str or Path
        Where to write the STL file.
    threshold : float
        Density isovalue. Default 0.5.
    smooth_passes : int
        Number of Taubin smooth iterations. Default 2.
    lamb : float
        Taubin forward smoothing factor. Default 0.5.
    nu : float
        Taubin backward anti-shrink factor. Default 0.53.

    Returns
    -------
    dict with keys:
        n_vertices, n_faces, volume_m3, bbox_mm (dict)
    """
    from skimage.measure import marching_cubes
    import trimesh
    import trimesh.smoothing

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    hx = mesh.lx / nx
    hy = mesh.ly / ny
    hz = mesh.lz / nz

    # 1. Reshape to 3D
    rho_3d = rho.reshape(nx, ny, nz)

    # 2. Pad with one layer of void on all 6 faces.
    # This guarantees the isosurface is always closed — marching cubes
    # cannot produce open boundaries if the field is void at every edge.
    # Without this, solid elements touching the domain boundary produce
    # holes (960 open edges, 24 holes in the test case).
    rho_padded = np.zeros((nx+2, ny+2, nz+2), dtype=np.float64)
    rho_padded[1:-1, 1:-1, 1:-1] = rho_3d

    # 3. Marching cubes on padded field
    verts, faces, _, _ = marching_cubes(
        rho_padded,
        level=threshold,
        spacing=(hx, hy, hz),
        allow_degenerate=False,
    )

    # 4. Shift vertices back by one voxel to undo the padding offset
    verts -= np.array([hx, hy, hz])

    if len(faces) == 0:
        raise ValueError(
            f"Marching cubes produced no faces at threshold={threshold}. "
            f"Try lowering the threshold or increasing the mesh resolution."
        )

    # 3. Taubin smoothing — removes staircase artifacts, preserves volume
    surf = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    # Taubin smoothing — more passes for grey fields
    trimesh.smoothing.filter_taubin(
        surf, lamb=0.5, nu=0.53, iterations=smooth_passes
    )

    # Keep only the largest connected component
    components = surf.split(only_watertight=False)
    if len(components) > 1:
        surf = max(components, key=lambda c: len(c.faces))

    # Clean up — use process() which runs all safe repairs
    surf.process(validate=True)
    surf.fix_normals()

    # 4. Export STL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surf.export(str(output_path), file_type='stl')

    # Bounding box in mm for the report
    bbox = surf.bounds  # shape (2, 3) in metres
    size_mm = (bbox[1] - bbox[0]) * 1000

    return {
        "n_vertices":  len(surf.vertices),
        "n_faces":     len(surf.faces),
        "volume_m3":   float(surf.volume),
        "bbox_mm":     {
            "x": float(size_mm[0]),
            "y": float(size_mm[1]),
            "z": float(size_mm[2]),
        },
        "stl_path":    str(output_path),
    }


def stl_to_base64(stl_path: str | Path) -> str:
    """Read an STL file and return its contents as a base64 string."""
    import base64
    with open(stl_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def base64_to_stl(b64: str, output_path: str | Path) -> None:
    """Decode a base64 string and write it as an STL file."""
    import base64
    data = base64.b64decode(b64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(data)


def stl_to_voxels(
    stl_path:   str | Path,
    nx: int, ny: int, nz: int,
    volume_fraction: float = 0.4,
    rho_min:    float = 0.001,
) -> tuple[np.ndarray, dict]:
    """
    Voxelise an STL into a design domain density field (Decision 16).

    Elements whose centres lie inside the STL solid are initialised at
    volume_fraction. Elements outside are set to rho_min. This defines
    the design domain — SIMP can only redistribute material inside the
    original geometry boundary.

    Parameters
    ----------
    stl_path : path to input STL (assumed millimetres)
    nx, ny, nz : voxel grid resolution
    volume_fraction : initial density for inside elements
    rho_min : density for outside elements

    Returns
    -------
    rho : np.ndarray, shape (nx*ny*nz,)
    domain_info : dict with lx, ly, lz (metres), voxel_mm
    """
    import trimesh

    mesh = trimesh.load(str(stl_path), force='mesh')

    # Bounding box in mm
    bbox   = mesh.bounds               # (2,3) in STL units (mm)
    size   = bbox[1] - bbox[0]         # (3,) in mm
    origin = bbox[0]                   # (3,) in mm

    lx_m = float(size[0]) / 1000.0
    ly_m = float(size[1]) / 1000.0
    lz_m = float(size[2]) / 1000.0

    hx_mm = float(size[0]) / nx
    hy_mm = float(size[1]) / ny
    hz_mm = float(size[2]) / nz

    # Centroid coordinates in mm for each voxel
    ci = (np.arange(nx) + 0.5) * hx_mm + origin[0]
    cj = (np.arange(ny) + 0.5) * hy_mm + origin[1]
    ck = (np.arange(nz) + 0.5) * hz_mm + origin[2]

    gi, gj, gk = np.meshgrid(ci, cj, ck, indexing='ij')
    centroids   = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1)

    # Inside/outside test using trimesh ray casting
    inside = mesh.contains(centroids)   # (n_elem,) bool

    rho = np.where(inside, volume_fraction, rho_min)

    domain_info = {
        "lx_m": lx_m, "ly_m": ly_m, "lz_m": lz_m,
        "voxel_mm": {"x": hx_mm, "y": hy_mm, "z": hz_mm},
        "bbox_mm": {
            "xmin": float(origin[0]), "xmax": float(bbox[1][0]),
            "ymin": float(origin[1]), "ymax": float(bbox[1][1]),
            "zmin": float(origin[2]), "zmax": float(bbox[1][2]),
        }
    }
    return rho, domain_info
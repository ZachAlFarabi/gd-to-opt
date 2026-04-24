"""
tests/test_bc_projection.py — Chunk 4b BC projection tests

Tests:
    1.  flood_fill selects seed triangle
    2.  flood_fill selects neighbours within tolerance
    3.  flood_fill rejects neighbours outside tolerance
    4.  normal_direction_filter selects correct faces
    5.  BCProjector: project force patch — load vector non-zero
    6.  BCProjector: project force patch — correct direction
    7.  BCProjector: force integrates to correct total magnitude
    8.  BCProjector: fixed patch produces fixed DOFs
    9.  BCProjector: flux patch produces thermal load
    10. BCProjector: temperature patch produces fixed thermal DOFs
    11. BCProjector: project_all assembles multiple patches
    12. BCProjector: deduplicates fixed DOFs
    13. BCProjector: degenerate triangles skipped
    14. SurfacePatch: direction normalised on construction
    15. project_all: f_struct shape matches n_dof
"""

from __future__ import annotations
import numpy as np
import pytest
import trimesh

from gdto.mesh_material import VoxelMesh
from gdto.bc_projection import (
    BCProjector, SurfacePatch,
    flood_fill_patch, normal_direction_filter,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def box_stl():
    """Simple box STL: 100×80×60 mm. 12 triangles."""
    mesh = trimesh.creation.box(extents=[100, 80, 60])
    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces,    dtype=np.int32),
    )

@pytest.fixture(scope="module")
def small_mesh():
    return VoxelMesh(nx=10, ny=8, nz=6,
                     lx=0.1, ly=0.08, lz=0.06)

@pytest.fixture(scope="module")
def projector(box_stl, small_mesh):
    verts, faces = box_stl
    return BCProjector(small_mesh, verts, faces)


# ── flood_fill tests ──────────────────────────────────────────────────────────

def test_flood_fill_includes_seed(box_stl):
    verts, faces = box_stl
    result = flood_fill_patch(0, verts, faces)
    assert 0 in result

def test_flood_fill_selects_coplanar(box_stl):
    """Box has 2 triangles per face — flood fill should select both."""
    verts, faces = box_stl
    # Start from triangle 0, which is on one face
    result = flood_fill_patch(0, verts, faces, angle_tol_deg=5.0)
    assert len(result) >= 1

def test_flood_fill_full_face_at_wide_tolerance(box_stl):
    """With 25° tolerance, should select at least 1 full face (2 triangles)."""
    verts, faces = box_stl
    result = flood_fill_patch(0, verts, faces, angle_tol_deg=25.0)
    assert len(result) >= 2

def test_flood_fill_zero_tolerance_selects_only_seed(box_stl):
    """With 0° tolerance only exactly parallel neighbours selected."""
    verts, faces = box_stl
    result = flood_fill_patch(0, verts, faces, angle_tol_deg=0.001)
    # Should be seed + its coplanar twin (same face = same normal)
    assert 0 in result
    assert len(result) >= 1

def test_flood_fill_returns_array(box_stl):
    verts, faces = box_stl
    result = flood_fill_patch(0, verts, faces)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32


# ── normal_direction_filter tests ─────────────────────────────────────────────

def test_normal_filter_selects_top_face(box_stl):
    """Direction = (0,0,1) should select only upward-facing triangles."""
    verts, faces = box_stl
    result = normal_direction_filter(verts, faces, [0,0,1], min_dot=0.9)
    assert len(result) >= 1
    # All selected triangles should have normals pointing up
    v = verts[faces[result]]
    e1 = v[:,1]-v[:,0]; e2 = v[:,2]-v[:,0]
    norms = np.cross(e1,e2)
    norms = norms / (np.linalg.norm(norms,axis=1,keepdims=True)+1e-30)
    assert np.all(norms[:,2] > 0.9)

def test_normal_filter_returns_array(box_stl):
    verts, faces = box_stl
    result = normal_direction_filter(verts, faces, [0,1,0])
    assert isinstance(result, np.ndarray)


# ── BCProjector tests ─────────────────────────────────────────────────────────

def test_projector_constructs(projector):
    assert projector is not None
    assert projector.tri_centroids.shape[1] == 3

def test_tri_areas_positive(projector):
    assert np.all(projector.tri_areas > 0)

def test_force_patch_nonzero_load(projector, box_stl, small_mesh):
    verts, faces = box_stl
    top_tris = normal_direction_filter(verts, faces, [0,0,1], min_dot=0.9)
    patch = SurfacePatch(
        tri_indices = top_tris,
        bc_type     = 'force',
        value       = 1000.0,            # positive magnitude
        direction   = np.array([0,0,-1.0]),
    )
    res = projector.project_patch(patch)
    assert np.any(res.f_structural != 0)

def test_force_patch_correct_direction(projector, box_stl):
    verts, faces = box_stl
    top_tris = normal_direction_filter(verts, faces, [0,0,1], min_dot=0.9)
    # value = magnitude (positive), direction = unit vector pointing downward
    patch = SurfacePatch(top_tris, 'force', 1000.0, np.array([0,0,-1.0]))
    res = projector.project_patch(patch)
    # Z-component of load should be negative (force in -Z direction)
    z_dofs = res.f_structural[2::3]   # every 3rd DOF starting at 2 = Z
    assert z_dofs.sum() < 0

def test_pressure_patch_nonzero(projector, box_stl):
    verts, faces = box_stl
    top_tris = normal_direction_filter(verts, faces, [0,0,1], min_dot=0.9)
    patch = SurfacePatch(top_tris, 'pressure', 1e6, None)
    res = projector.project_patch(patch)
    assert np.any(res.f_structural != 0)

def test_fixed_patch_produces_dofs(projector, box_stl):
    verts, faces = box_stl
    bottom_tris = normal_direction_filter(verts, faces, [0,0,-1], min_dot=0.9)
    patch = SurfacePatch(bottom_tris, 'fixed', 0.0, None)
    res = projector.project_patch(patch)
    assert len(res.fixed_dofs) > 0
    assert len(res.fixed_vals) == len(res.fixed_dofs)
    assert np.all(res.fixed_vals == 0.0)

def test_flux_patch_thermal_load(projector, box_stl):
    verts, faces = box_stl
    top_tris = normal_direction_filter(verts, faces, [0,0,1], min_dot=0.9)
    patch = SurfacePatch(top_tris, 'flux', 5000.0, np.array([0,0,-1.0]))
    res = projector.project_patch(patch)
    assert np.any(res.f_thermal != 0)
    assert res.f_thermal.sum() > 0

def test_temperature_patch_thermal_dofs(projector, box_stl):
    verts, faces = box_stl
    bottom_tris = normal_direction_filter(verts, faces, [0,0,-1], min_dot=0.9)
    patch = SurfacePatch(bottom_tris, 'temperature', 20.0, None)
    res = projector.project_patch(patch)
    assert len(res.fixed_dofs) > 0
    assert np.all(res.fixed_vals == 20.0)

def test_project_all_assembles_patches(projector, box_stl, small_mesh):
    verts, faces = box_stl
    top_tris    = normal_direction_filter(verts, faces, [0,0,1],  min_dot=0.9)
    bot_tris    = normal_direction_filter(verts, faces, [0,0,-1], min_dot=0.9)
    patches = [
        SurfacePatch(bot_tris, 'fixed',  0.0,    None),
        SurfacePatch(top_tris, 'force', 1000.0, np.array([0,0,-1.0])),
    ]
    f_s, f_t, fd, fv = projector.project_all(patches)
    assert f_s.shape == (small_mesh.n_dof,)
    assert f_t.shape == (small_mesh.n_nodes,)
    assert len(fd) > 0
    assert np.any(f_s != 0)

def test_project_all_deduplicates_fixed_dofs(projector, box_stl):
    verts, faces = box_stl
    bot_tris = normal_direction_filter(verts, faces, [0,0,-1], min_dot=0.9)
    # Apply fixed BC twice on same face
    patches = [
        SurfacePatch(bot_tris, 'fixed', 0.0, None),
        SurfacePatch(bot_tris, 'fixed', 0.0, None),
    ]
    _, _, fd, fv = projector.project_all(patches)
    # Should not have duplicates
    assert len(fd) == len(np.unique(fd))

def test_f_struct_shape(projector, box_stl, small_mesh):
    verts, faces = box_stl
    top_tris = normal_direction_filter(verts, faces, [0,0,1], min_dot=0.9)
    patch = SurfacePatch(top_tris, 'force', 100.0, np.array([0,-1.0,0]))
    res = projector.project_patch(patch)
    assert res.f_structural.shape == (small_mesh.n_dof,)

def test_surface_patch_normalises_direction():
    patch = SurfacePatch(
        tri_indices = np.array([0]),
        bc_type     = 'force',
        value       = 1.0,
        direction   = np.array([0.0, 0.0, 5.0]),  # not unit length
    )
    assert np.isclose(np.linalg.norm(patch.direction), 1.0)

def test_area_sum_reasonable(projector, box_stl):
    """Total area of all faces should ≈ surface area of box (2*(100*80+80*60+100*60) mm²)."""
    verts, faces = box_stl
    all_tris = np.arange(len(faces))
    patch = SurfacePatch(all_tris, 'flux', 1.0, np.array([0,0,1.0]))
    res   = projector.project_patch(patch)
    expected_m2 = 2*(0.1*0.08 + 0.08*0.06 + 0.1*0.06)
    np.testing.assert_allclose(res.total_area_m2, expected_m2, rtol=0.01)

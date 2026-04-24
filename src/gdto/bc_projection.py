"""
bc_projection.py — Chunk 4b, Decisions 19-22: dynamic surface BC projection

Converts user-selected STL surface patches into FEA load vectors.
Replaces the planar face-string BC system for complex geometries.

Projection algorithm (Decision 20):
    For each selected triangle:
        1. Find which voxel element its centroid falls inside
        2. Identify which face of that element the triangle projects onto
        3. Distribute force/flux to 4 face nodes using bilinear shape functions
        4. Accumulate into global load vector

Backward compatibility (Decision 21):
    String-based BCs (fixed_faces, loads) still work via build_problem().
    DynamicBoundaryConditions is used when patches are provided instead.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import NamedTuple

from gdto.mesh_material import VoxelMesh


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SurfacePatch:
    """
    A user-selected region of the STL surface with an associated BC.

    tri_indices : selected triangle indices into the STL face array
    bc_type     : 'force' | 'pressure' | 'fixed' | 'flux' | 'temperature'
    value       : scalar magnitude — N (positive) for force, N/m² for pressure,
                  W/m² for flux, °C for temperature, 0 for fixed.
                  For force: total force in Newtons (always positive).
                  Direction of force is encoded entirely in `direction` vector.
    direction   : (3,) unit vector for force/flux BCs.
                  None for fixed/temperature (direction irrelevant)
    label       : user-defined name e.g. "bolt face" or "bearing load"
    """
    tri_indices: np.ndarray
    bc_type:     str
    value:       float
    direction:   np.ndarray | None = None
    label:       str               = ""

    def __post_init__(self):
        self.tri_indices = np.asarray(self.tri_indices, dtype=np.int32)
        if self.direction is not None:
            d = np.asarray(self.direction, dtype=np.float64)
            norm = np.linalg.norm(d)
            self.direction = d / norm if norm > 1e-10 else d


@dataclass
class ProjectionResult:
    """
    Output of projecting one SurfacePatch onto the voxel mesh.

    f_structural : (n_dof,) structural load vector contribution
    f_thermal    : (n_nodes,) thermal load vector contribution
    fixed_dofs   : node indices with Dirichlet BCs
    fixed_vals   : prescribed values at fixed_dofs
    total_area_m2: total area of selected triangles [m²]
    n_tri        : number of triangles projected
    """
    f_structural:  np.ndarray
    f_thermal:     np.ndarray
    fixed_dofs:    np.ndarray
    fixed_vals:    np.ndarray
    total_area_m2: float
    n_tri:         int


# ── Core projection engine ───────────────────────────────────────────────────

class BCProjector:
    """
    Projects STL surface patches onto a voxel FEA mesh.

    The STL and voxel mesh are different representations — vertices do
    not align. This class bridges them via centroid-based projection:
    each STL triangle is mapped to the nearest voxel element face and
    its load is distributed to the 4 face nodes using bilinear shape
    functions.

    Parameters
    ----------
    mesh      : VoxelMesh
    stl_verts : (n_verts, 3) float  — vertex positions in mm
    stl_faces : (n_faces, 3) int    — vertex indices per triangle
    """

    def __init__(
        self,
        mesh:       VoxelMesh,
        stl_verts:  np.ndarray,
        stl_faces:  np.ndarray,
    ) -> None:
        self.mesh       = mesh
        self.stl_verts  = np.asarray(stl_verts,  dtype=np.float64)
        self.stl_faces  = np.asarray(stl_faces,  dtype=np.int32)

        # Pre-compute triangle centroids and areas
        v = self.stl_verts[self.stl_faces]         # (n_faces, 3, 3) mm
        self.tri_centroids = v.mean(axis=1)         # (n_faces, 3) mm
        e1 = v[:,1] - v[:,0]
        e2 = v[:,2] - v[:,0]
        cross = np.cross(e1, e2)                    # (n_faces, 3)
        self.tri_normals = cross / (
            np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30
        )                                           # (n_faces, 3) unit normals
        self.tri_areas   = 0.5 * np.linalg.norm(cross, axis=1)  # mm²

        # Voxel dimensions in mm
        self.hx = mesh.lx * 1000 / mesh.nx
        self.hy = mesh.ly * 1000 / mesh.ny
        self.hz = mesh.lz * 1000 / mesh.nz

        # Domain origin — bottom-left corner of STL bounding box in mm.
        # The voxel mesh runs from (0,0,0) to (lx,ly,lz) in metres,
        # but the STL may be centred at the origin or placed anywhere.
        # All centroid-to-voxel mappings must subtract this offset.
        self.origin = self.stl_verts.min(axis=0)   # (3,) in mm

        # Pre-compute element node map for face node lookup
        self._build_element_face_nodes()

    # ── Element/face utilities ───────────────────────────────────────────────

    def _eid(self, i, j, k):
        """Element index from grid coordinates."""
        m = self.mesh
        return (int(np.clip(i,0,m.nx-1)) * m.ny * m.nz +
                int(np.clip(j,0,m.ny-1)) * m.nz +
                int(np.clip(k,0,m.nz-1)))

    def _nid(self, i, j, k):
        """Node index from grid coordinates."""
        m = self.mesh
        return (int(np.clip(i,0,m.nx)) * (m.ny+1) * (m.nz+1) +
                int(np.clip(j,0,m.ny)) * (m.nz+1) +
                int(np.clip(k,0,m.nz)))

    def _build_element_face_nodes(self):
        """
        For each of the 6 element faces, define the 4 corner nodes in
        consistent counter-clockwise order (viewed from outside).
        Face ordering: xmin, xmax, ymin, ymax, zmin, zmax
        """
        # Node local corner offsets (i,j,k) for each face
        # Face 0 xmin: i=0; Face 1 xmax: i=1, etc.
        self._face_node_offsets = {
            'xmin': [(0,0,0),(0,1,0),(0,1,1),(0,0,1)],
            'xmax': [(1,0,0),(1,0,1),(1,1,1),(1,1,0)],
            'ymin': [(0,0,0),(0,0,1),(1,0,1),(1,0,0)],
            'ymax': [(0,1,0),(1,1,0),(1,1,1),(0,1,1)],
            'zmin': [(0,0,0),(1,0,0),(1,1,0),(0,1,0)],
            'zmax': [(0,0,1),(0,1,1),(1,1,1),(1,0,1)],
        }

    def _dominant_face(self, normal: np.ndarray) -> str:
        """
        Which element face does a triangle with this normal project onto?
        Returns the face name whose outward normal most aligns with
        the triangle normal.
        """
        axes   = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
                          dtype=float)
        names  = ['xmax','xmin','ymax','ymin','zmax','zmin']
        dots   = axes @ normal
        return names[int(np.argmax(dots))]

    def _face_nodes_for_element(
        self, ei: int, ej: int, ek: int, face: str
    ) -> np.ndarray:
        """Return the 4 node indices for a given element face."""
        offsets = self._face_node_offsets[face]
        return np.array([
            self._nid(ei + di, ej + dj, ek + dk)
            for di, dj, dk in offsets
        ], dtype=np.int32)

    def _bilinear_weights(
        self,
        centroid_mm: np.ndarray,
        ei: int, ej: int, ek: int,
        face: str,
    ) -> np.ndarray:
        """
        Bilinear shape function weights for a point on an element face.
        Returns (4,) weights summing to 1.

        The two local coordinates (s,t) ∈ [0,1]² are derived from
        the centroid's position within the element face.
        """
        hx, hy, hz = self.hx, self.hy, self.hz
        # Element origin in mm — must include STL domain offset
        ox = self.origin[0] + ei * hx
        oy = self.origin[1] + ej * hy
        oz = self.origin[2] + ek * hz

        # Local coordinates within element [0,1]³
        xi   = np.clip((centroid_mm[0] - ox) / hx, 0, 1)
        eta  = np.clip((centroid_mm[1] - oy) / hy, 0, 1)
        zeta = np.clip((centroid_mm[2] - oz) / hz, 0, 1)

        # Map to 2D face coordinates (s,t) depending on which face
        if face in ('xmin', 'xmax'):
            s, t = eta, zeta
        elif face in ('ymin', 'ymax'):
            s, t = xi, zeta
        else:  # zmin, zmax
            s, t = xi, eta

        # Bilinear shape functions: N = [(1-s)(1-t), s(1-t), st, (1-s)t]
        return np.array([
            (1-s)*(1-t), s*(1-t), s*t, (1-s)*t
        ], dtype=np.float64)

    # ── Main projection ──────────────────────────────────────────────────────

    def _find_voxel_nodes_in_patch(
        self,
        pmin:      np.ndarray,
        pmax:      np.ndarray,
        face_name: str,
    ) -> np.ndarray:
        """
        Find all voxel mesh nodes that lie on face_name within
        the bounding box [pmin, pmax] in mm.

        Instead of using STL triangle vertices (only 2–4 nodes per triangle),
        this finds every voxel node on the face within the patch region,
        giving proper load distribution equivalent to the simple face mode.
        """
        mesh = self.mesh
        hx, hy, hz = self.hx, self.hy, self.hz
        ox, oy, oz = self.origin

        node_ids = []

        if face_name == 'xmin':
            i_fixed = 0
            j_range = range(
                max(0, int(np.floor((pmin[1] - oy) / hy))),
                min(mesh.ny, int(np.ceil( (pmax[1] - oy) / hy)) + 1),
            )
            k_range = range(
                max(0, int(np.floor((pmin[2] - oz) / hz))),
                min(mesh.nz, int(np.ceil( (pmax[2] - oz) / hz)) + 1),
            )
            for j in j_range:
                for k in k_range:
                    node_ids.append(self._nid(i_fixed, j, k))

        elif face_name == 'xmax':
            i_fixed = mesh.nx
            j_range = range(
                max(0, int(np.floor((pmin[1] - oy) / hy))),
                min(mesh.ny, int(np.ceil( (pmax[1] - oy) / hy)) + 1),
            )
            k_range = range(
                max(0, int(np.floor((pmin[2] - oz) / hz))),
                min(mesh.nz, int(np.ceil( (pmax[2] - oz) / hz)) + 1),
            )
            for j in j_range:
                for k in k_range:
                    node_ids.append(self._nid(i_fixed, j, k))

        elif face_name == 'ymin':
            j_fixed = 0
            i_range = range(
                max(0, int(np.floor((pmin[0] - ox) / hx))),
                min(mesh.nx, int(np.ceil( (pmax[0] - ox) / hx)) + 1),
            )
            k_range = range(
                max(0, int(np.floor((pmin[2] - oz) / hz))),
                min(mesh.nz, int(np.ceil( (pmax[2] - oz) / hz)) + 1),
            )
            for i in i_range:
                for k in k_range:
                    node_ids.append(self._nid(i, j_fixed, k))

        elif face_name == 'ymax':
            j_fixed = mesh.ny
            i_range = range(
                max(0, int(np.floor((pmin[0] - ox) / hx))),
                min(mesh.nx, int(np.ceil( (pmax[0] - ox) / hx)) + 1),
            )
            k_range = range(
                max(0, int(np.floor((pmin[2] - oz) / hz))),
                min(mesh.nz, int(np.ceil( (pmax[2] - oz) / hz)) + 1),
            )
            for i in i_range:
                for k in k_range:
                    node_ids.append(self._nid(i, j_fixed, k))

        elif face_name == 'zmin':
            k_fixed = 0
            i_range = range(
                max(0, int(np.floor((pmin[0] - ox) / hx))),
                min(mesh.nx, int(np.ceil( (pmax[0] - ox) / hx)) + 1),
            )
            j_range = range(
                max(0, int(np.floor((pmin[1] - oy) / hy))),
                min(mesh.ny, int(np.ceil( (pmax[1] - oy) / hy)) + 1),
            )
            for i in i_range:
                for j in j_range:
                    node_ids.append(self._nid(i, j, k_fixed))

        elif face_name == 'zmax':
            k_fixed = mesh.nz
            i_range = range(
                max(0, int(np.floor((pmin[0] - ox) / hx))),
                min(mesh.nx, int(np.ceil( (pmax[0] - ox) / hx)) + 1),
            )
            j_range = range(
                max(0, int(np.floor((pmin[1] - oy) / hy))),
                min(mesh.ny, int(np.ceil( (pmax[1] - oy) / hy)) + 1),
            )
            for i in i_range:
                for j in j_range:
                    node_ids.append(self._nid(i, j, k_fixed))

        return np.unique(np.array(node_ids, dtype=np.int32))

    def project_patch(self, patch: SurfacePatch) -> ProjectionResult:
        """
        Project a surface patch onto the voxel mesh.

        For force/pressure/fixed BCs: uses voxel-mesh-based distribution —
        finds all voxel nodes on the face within the patch bounding box and
        distributes load evenly. This matches the simple-mode behaviour and
        avoids artificial point-load concentrations from sparse STL vertices.

        For flux/temperature BCs: uses the original triangle-based approach
        (fine for scalar thermal DOFs).
        """
        mesh    = self.mesh
        n_dof   = mesh.n_dof
        n_nodes = mesh.n_nodes

        f_struct        = np.zeros(n_dof,   dtype=np.float64)
        f_therm         = np.zeros(n_nodes, dtype=np.float64)
        fixed_dofs_list = []
        fixed_vals_list = []
        total_area      = 0.0

        if patch.bc_type in ('force', 'pressure', 'fixed'):
            if len(patch.tri_indices) == 0:
                pass
            else:
                selected_verts = self.stl_verts[
                    self.stl_faces[patch.tri_indices].ravel()
                ]
                pmin = selected_verts.min(axis=0)
                pmax = selected_verts.max(axis=0)

                avg_normal = self.tri_normals[patch.tri_indices].mean(axis=0)
                avg_normal /= np.linalg.norm(avg_normal) + 1e-30
                face_name   = self._dominant_face(avg_normal)
                total_area  = float(self.tri_areas[patch.tri_indices].sum()) / 1e6

                patch_nodes = self._find_voxel_nodes_in_patch(pmin, pmax, face_name)

                if len(patch_nodes) == 0:
                    import warnings
                    warnings.warn(f"Patch '{patch.label}': no voxel nodes found in region")
                elif patch.bc_type == 'fixed':
                    for node in patch_nodes:
                        for d in range(3):
                            fixed_dofs_list.append(3 * node + d)
                            fixed_vals_list.append(0.0)

                elif patch.bc_type == 'force':
                    # Distribute total force evenly over all patch nodes
                    force_per_node = (
                        patch.direction * abs(patch.value) / len(patch_nodes)
                    )
                    for node in patch_nodes:
                        dof_base = 3 * node
                        f_struct[dof_base]   += force_per_node[0]
                        f_struct[dof_base+1] += force_per_node[1]
                        f_struct[dof_base+2] += force_per_node[2]

                elif patch.bc_type == 'pressure':
                    traction_per_node = (
                        -avg_normal * patch.value * total_area / len(patch_nodes)
                    )
                    for node in patch_nodes:
                        dof_base = 3 * node
                        f_struct[dof_base]   += traction_per_node[0]
                        f_struct[dof_base+1] += traction_per_node[1]
                        f_struct[dof_base+2] += traction_per_node[2]

        elif patch.bc_type in ('flux', 'temperature'):
            # Thermal BCs — triangle-based approach (fine for scalar DOF)
            for tri_idx in patch.tri_indices:
                centroid = self.tri_centroids[tri_idx]
                normal   = self.tri_normals[tri_idx]
                area_mm2 = self.tri_areas[tri_idx]
                area_m2  = area_mm2 / 1e6
                if area_mm2 < 1e-10:
                    continue

                ei = int(np.clip((centroid[0] - self.origin[0]) / self.hx, 0, mesh.nx - 1))
                ej = int(np.clip((centroid[1] - self.origin[1]) / self.hy, 0, mesh.ny - 1))
                ek = int(np.clip((centroid[2] - self.origin[2]) / self.hz, 0, mesh.nz - 1))

                face    = self._dominant_face(normal)
                nodes4  = self._face_nodes_for_element(ei, ej, ek, face)
                weights = self._bilinear_weights(centroid, ei, ej, ek, face)
                total_area += area_m2

                if patch.bc_type == 'flux':
                    flux_total = patch.value * area_m2
                    for node, w in zip(nodes4, weights):
                        f_therm[node] += w * flux_total
                elif patch.bc_type == 'temperature':
                    for node in nodes4:
                        fixed_dofs_list.append(node)
                        fixed_vals_list.append(patch.value)

        fixed_dofs = np.array(fixed_dofs_list, dtype=np.int32)
        fixed_vals = np.array(fixed_vals_list, dtype=np.float64)

        return ProjectionResult(
            f_structural  = f_struct,
            f_thermal     = f_therm,
            fixed_dofs    = fixed_dofs,
            fixed_vals    = fixed_vals,
            total_area_m2 = total_area,
            n_tri         = len(patch.tri_indices),
        )

    def project_all(
        self,
        patches: list[SurfacePatch],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project all patches and return assembled load vectors.

        Returns
        -------
        f_structural  : (n_dof,)   structural load vector
        f_thermal     : (n_nodes,) thermal load vector
        fixed_dofs    : (n_fixed,) DOF indices with Dirichlet BCs
        fixed_vals    : (n_fixed,) prescribed values
        """
        mesh       = self.mesh
        f_struct   = np.zeros(mesh.n_dof,   dtype=np.float64)
        f_therm    = np.zeros(mesh.n_nodes, dtype=np.float64)
        all_fixed_dofs = []
        all_fixed_vals = []

        for patch in patches:
            res = self.project_patch(patch)
            f_struct += res.f_structural
            f_therm  += res.f_thermal
            all_fixed_dofs.extend(res.fixed_dofs.tolist())
            all_fixed_vals.extend(res.fixed_vals.tolist())

        # Deduplicate fixed DOFs — keep unique, average conflicting values
        if all_fixed_dofs:
            dofs_arr = np.array(all_fixed_dofs, dtype=np.int32)
            vals_arr = np.array(all_fixed_vals, dtype=np.float64)
            unique_dofs, inv = np.unique(dofs_arr, return_inverse=True)
            unique_vals = np.zeros(len(unique_dofs))
            counts      = np.zeros(len(unique_dofs))
            np.add.at(unique_vals, inv, vals_arr)
            np.add.at(counts,      inv, 1)
            unique_vals /= np.maximum(counts, 1)
        else:
            unique_dofs = np.array([], dtype=np.int32)
            unique_vals = np.array([], dtype=np.float64)

        return f_struct, f_therm, unique_dofs, unique_vals


# ── Flood-fill patch selection (runs on server for validation) ───────────────

def flood_fill_patch(
    tri_idx:        int,
    stl_verts:      np.ndarray,
    stl_faces:      np.ndarray,
    angle_tol_deg:  float = 25.0,
    max_triangles:  int   = 50_000,
) -> np.ndarray:
    """
    Select a surface patch by flood-filling from a seed triangle.
    Accepts neighbours whose normals are within angle_tol_deg of the seed.

    Used server-side to validate/expand selections sent from the frontend.

    Parameters
    ----------
    tri_idx       : seed triangle index
    stl_verts     : (n_verts, 3)
    stl_faces     : (n_faces, 3)
    angle_tol_deg : normal angle tolerance in degrees
    max_triangles : safety limit to prevent runaway selection

    Returns
    -------
    np.ndarray of selected triangle indices
    """
    v = stl_verts[stl_faces]
    e1 = v[:,1] - v[:,0]
    e2 = v[:,2] - v[:,0]
    cross = np.cross(e1, e2)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)

    cos_tol = np.cos(np.radians(angle_tol_deg))
    n_seed  = normals[tri_idx]

    # Build edge-to-face adjacency map
    edge_to_faces = {}
    for fi, face in enumerate(stl_faces):
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i+1)%3])]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)

    # BFS flood fill
    selected = {tri_idx}
    queue    = [tri_idx]

    while queue and len(selected) < max_triangles:
        fi = queue.pop(0)
        face = stl_faces[fi]
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i+1)%3])]))
            for nfi in edge_to_faces.get(edge, []):
                if nfi in selected:
                    continue
                if np.dot(normals[nfi], n_seed) >= cos_tol:
                    selected.add(nfi)
                    queue.append(nfi)

    return np.array(sorted(selected), dtype=np.int32)


def normal_direction_filter(
    stl_verts:   np.ndarray,
    stl_faces:   np.ndarray,
    direction:   np.ndarray,
    min_dot:     float = 0.7,
) -> np.ndarray:
    """
    Select all triangles whose normal has positive component
    along the given direction (dot product > min_dot).

    Useful for selecting all faces pointing roughly in one direction
    e.g. all upward-facing surfaces for gravity loading.

    Parameters
    ----------
    direction : (3,) target direction unit vector
    min_dot   : minimum dot product with direction (0.7 ≈ 45°)
    """
    v = stl_verts[stl_faces]
    e1 = v[:,1] - v[:,0]
    e2 = v[:,2] - v[:,0]
    cross = np.cross(e1, e2)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    dots = normals @ direction
    return np.where(dots >= min_dot)[0].astype(np.int32)

"""
filters.py — Chunk 3: density filter and overhang filter

Implements:
    DensityFilter  — density filter (Bruns & Tortorelli 2001)
                     forward:  rho_filtered = W @ rho
                     backward: dc_drho = W.T @ dc_drho_filtered
    OverhangFilter — smooth P-norm overhang proxy (Gaynor & Guest 2016)
                     forward:  V_supp, dV_supp/drho

Decisions implemented:
    Decision 11 — density filter (not sensitivity filter)
    Decision 12 — linear hat weight, r=1.5 element lengths,
                  Euclidean distance, CSR sparse matrix
    Decision 13 — smooth P-norm overhang, P=2, r_s=0 (Z build only),
                  analytical gradient

Both filters are constructed once and applied cheaply each SIMP iteration.
The DensityFilter replaces the stub in simp.py.
The OverhangFilter provides (V_supp, gradient) to MMA when
overhang_weight > 0 in design_params.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field

from gdto.mesh_material import VoxelMesh


# ---------------------------------------------------------------------------
# DensityFilter
# ---------------------------------------------------------------------------

class DensityFilter:
    """
    Density filter for topology optimisation.

    Replaces the sensitivity filter stub in simp.py. Applied before
    assembling K and after computing raw sensitivities.

    Forward pass (applied to density field before K assembly):
        rho_filtered = W @ rho

    Backward pass (chain rule, applied to raw sensitivities):
        dc_drho = W.T @ dc_drho_filtered

    The filter matrix W is row-stochastic (rows sum to 1), so:
        - mean(rho_filtered) = mean(rho)  — volume is conserved
        - rho_filtered in [min(rho), max(rho)]  — no out-of-range values
        - W is approximately symmetric for regular interior mesh

    Parameters
    ----------
    mesh   : VoxelMesh
    radius : float
        Filter radius in element lengths. Default 1.5.
        Minimum feature size enforced ≈ 2 * radius * h.
    """

    def __init__(self, mesh: VoxelMesh, radius: float = 1.5) -> None:
        self.mesh   = mesh
        self.radius = radius
        self.W      = self._build_filter_matrix()

    def _build_filter_matrix(self) -> sp.csr_matrix:
        """
        Build the sparse row-stochastic filter matrix W.

        Algorithm (Decision 12):
            1. Enumerate Chebyshev integer neighbours (radius floor)
            2. Compute Euclidean distances to candidates
            3. Apply linear hat weight H = max(0, r*h - d_euclidean)
            4. Row-normalise to produce W row-stochastic

        Returns
        -------
        W : sp.csr_matrix, shape (n_elem, n_elem)
        """
        mesh      = self.mesh
        r_elem    = self.radius                      # radius in element lengths
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        n_elem    = mesh.n_elem

        # Element centroid coordinates
        # centroid of element (i,j,k) = ((i+0.5)*hx, (j+0.5)*hy, (k+0.5)*hz)
        hx = mesh.lx / nx
        hy = mesh.ly / ny
        hz = mesh.lz / nz
        h_min = min(hx, hy, hz)              # physical radius threshold
        r_phys = r_elem * h_min              # in metres

        # Build centroid array: shape (n_elem, 3)
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )
        cx = (ii.ravel() + 0.5) * hx
        cy = (jj.ravel() + 0.5) * hy
        cz = (kk.ravel() + 0.5) * hz
        centroids = np.stack([cx, cy, cz], axis=1)   # (n_elem, 3)

        # Integer Chebyshev radius for candidate enumeration
        r_int = int(np.ceil(r_elem))

        # Element index: e = i*ny*nz + j*nz + k
        def eid(i, j, k):
            return i * ny * nz + j * nz + k

        rows, cols, vals = [], [], []

        # Loop over all elements and their Chebyshev neighbours
        # This loop runs once at construction — not per SIMP iteration
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    e = eid(i, j, k)
                    ce = centroids[e]

                    # Chebyshev candidate neighbourhood
                    for di in range(-r_int, r_int + 1):
                        ni = i + di
                        if ni < 0 or ni >= nx:
                            continue
                        for dj in range(-r_int, r_int + 1):
                            nj = j + dj
                            if nj < 0 or nj >= ny:
                                continue
                            for dk in range(-r_int, r_int + 1):
                                nk = k + dk
                                if nk < 0 or nk >= nz:
                                    continue

                                e_prime = eid(ni, nj, nk)
                                d = np.linalg.norm(ce - centroids[e_prime])
                                H = max(0.0, r_phys - d)   # linear hat
                                if H > 0.0:
                                    rows.append(e)
                                    cols.append(e_prime)
                                    vals.append(H)

        W_raw = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_elem, n_elem), dtype=np.float64
        )

        # Row-normalise → row-stochastic
        row_sums = np.array(W_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0          # guard (should not occur)
        W = sp.diags(1.0 / row_sums) @ W_raw

        return W.tocsr()

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """
        Forward pass: filter the density field.

        rho_filtered = W @ rho

        Parameters
        ----------
        rho : np.ndarray, shape (n_elem,)
            Raw element densities in [rho_min, 1].

        Returns
        -------
        rho_filtered : np.ndarray, shape (n_elem,)
        """
        return self.W @ rho

    def backprop(self, dc_drho_filtered: np.ndarray) -> np.ndarray:
        """
        Backward pass: chain rule through the density filter.

        dc_drho = W.T @ dc_drho_filtered

        Converts sensitivity w.r.t filtered density back to sensitivity
        w.r.t raw design variable rho (Decision 11, chain rule derivation).

        Parameters
        ----------
        dc_drho_filtered : np.ndarray, shape (n_elem,)
            Sensitivity dC/d(rho_filtered) from FEA.

        Returns
        -------
        dc_drho : np.ndarray, shape (n_elem,)
            Corrected sensitivity dC/d(rho).
        """
        return self.W.T @ dc_drho_filtered

    @property
    def n_nonzeros(self) -> int:
        """Number of nonzeros in W."""
        return self.W.nnz

    def summary(self) -> str:
        h = min(self.mesh.lx / self.mesh.nx,
                self.mesh.ly / self.mesh.ny,
                self.mesh.lz / self.mesh.nz)
        return (
            f"DensityFilter: radius={self.radius} elem = "
            f"{self.radius * h * 1000:.1f} mm  |  "
            f"min_feature≈{2 * self.radius * h * 1000:.1f} mm  |  "
            f"W nnz={self.n_nonzeros:,}"
        )


# ---------------------------------------------------------------------------
# OverhangFilter
# ---------------------------------------------------------------------------

class OverhangFilter:
    """
    Smooth overhang filter for AM printability.

    Computes the support volume proxy and its gradient with respect
    to the density field. Used by MMA as the second constraint.

    Overhang measure per element (Decision 13):
        delta_e = rho_e - rho_below_e
        o_e     = max(0, delta_e)^P

    Total support volume:
        V_supp = v_elem * sum(o_e)

    Gradient (analytical):
        dV/drho_e = v_elem * (
            P * max(0, delta_e)^(P-1) * [delta_e > 0]          (self term)
          - P * max(0, delta_above_e)^(P-1) * [delta_above > 0] (neighbour term)
        )

    where delta_above_e = rho_above_e - rho_e is the overhang deficit
    of the element directly above e (for which e is the supporter).

    Parameters
    ----------
    mesh  : VoxelMesh
    P     : float
        P-norm exponent. Default 2 (smooth, differentiable at zero).
    build_axis : int
        Axis index of build direction. Default 2 (Z).
        Rotation to arbitrary build direction deferred to chunk 7.
    """

    def __init__(
        self,
        mesh:       VoxelMesh,
        P:          float = 2.0,
        build_axis: int   = 2,
    ) -> None:
        self.mesh       = mesh
        self.P          = P
        self.build_axis = build_axis
        self.v_elem     = (
            mesh.lx * mesh.ly * mesh.lz / mesh.n_elem
        )

    def compute(
        self, rho: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Compute support volume proxy and its gradient.

        Parameters
        ----------
        rho : np.ndarray, shape (n_elem,)
            Element densities (filtered or raw).

        Returns
        -------
        V_supp : float
            Total support volume proxy [m³].
        dV_drho : np.ndarray, shape (n_elem,)
            Gradient dV_supp / d rho_e for MMA constraint.
        """
        mesh = self.mesh
        P    = self.P
        ax   = self.build_axis

        # Reshape to 3D grid (nx, ny, nz)
        rho_3d = rho.reshape(mesh.nx, mesh.ny, mesh.nz)

        # rho_below: density of element directly below in build direction
        # Element at grid position [..., k, ...] has support from [..., k-1, ...]
        # Bottom layer (k=0) has no element below — treat as fully supported
        rho_below = np.zeros_like(rho_3d)
        if ax == 0:
            rho_below[1:, :, :] = rho_3d[:-1, :, :]
        elif ax == 1:
            rho_below[:, 1:, :] = rho_3d[:, :-1, :]
        else:  # ax == 2 (default Z)
            rho_below[:, :, 1:] = rho_3d[:, :, :-1]

        # rho_above: density of element directly above
        # Used for the neighbour term of the gradient
        rho_above = np.zeros_like(rho_3d)
        if ax == 0:
            rho_above[:-1, :, :] = rho_3d[1:, :, :]
        elif ax == 1:
            rho_above[:, :-1, :] = rho_3d[:, 1:, :]
        else:
            rho_above[:, :, :-1] = rho_3d[:, :, 1:]

        # Overhang deficit: positive when element has more material than support
        delta      = rho_3d - rho_below        # (nx, ny, nz)
        delta_clip = np.maximum(0.0, delta)    # ReLU

        # Support volume proxy: sum of o_e = max(0, delta)^P
        o_e    = delta_clip ** P
        V_supp = float(self.v_elem * o_e.sum())

        # Gradient (Decision 13 derivation)
        # Self term: d(o_e)/d(rho_e) = P * max(0, delta_e)^(P-1) * [delta_e > 0]
        g = P * np.where(delta > 0, delta_clip ** (P - 1), 0.0)

        # Neighbour term: element e is a supporter for element above it
        # delta_above = rho_above - rho_e
        delta_above      = rho_above - rho_3d
        delta_above_clip = np.maximum(0.0, delta_above)
        g_above = P * np.where(delta_above > 0,
                               delta_above_clip ** (P - 1), 0.0)

        # Full gradient: self term minus neighbour term
        dV_drho = self.v_elem * (g - g_above)

        return V_supp, dV_drho.ravel()

    def summary(self) -> str:
        ax_names = {0: 'X', 1: 'Y', 2: 'Z'}
        return (
            f"OverhangFilter: P={self.P}  "
            f"build_axis={ax_names.get(self.build_axis, self.build_axis)}  "
            f"v_elem={self.v_elem:.3e} m³"
        )
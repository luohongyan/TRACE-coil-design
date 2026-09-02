"""Axis-specific cylindrical FEM/Biot--Savart setup used by TRACE."""
from __future__ import annotations

import numpy as np

from biotsavart import build_coupling
from gradient_topopt import GY
from solver import Problem as YProblem


X_WIRING = (1, -1, 1)
Y_WIRING = (-1, 1, 1)
Z_WIRING = (-1, -1, -1)

WIRING = {"x": X_WIRING, "y": Y_WIRING, "z": Z_WIRING}
THETA_OFFSET = {"x": -0.5 * np.pi, "y": 0.0, "z": 0.0}


class PhaseAlignedXProblem(YProblem):
    """X-gradient problem obtained by rotating the Y setup by -90 degrees."""

    def __init__(self, theta_offset=-0.5 * np.pi, wiring=X_WIRING,
                 **kwargs):
        self.wiring = tuple(wiring)
        self.theta_offset = float(theta_offset)
        super().__init__(**kwargs)
        self.C = build_coupling(
            self.mesh, self.roi, self.wiring,
            theta_offset=self.theta_offset,
        )
        self.Bzobj = GY * self.roi[:, 0]


class ZProblem(YProblem):
    """Z-gradient problem with circumferential drive and odd mirror wiring."""

    def _setup_bc(self, d1_frac):
        del d1_frac
        mesh = self.mesh
        self.D1 = np.array([
            mesh.node_index(0, j) for j in range(mesh.nz + 1)
        ])
        self.D2 = np.array([
            mesh.node_index(mesh.nl, j) for j in range(mesh.nz + 1)
        ])
        self.fixed = np.unique(np.concatenate([self.D1, self.D2]))
        self.free = np.setdiff1d(np.arange(mesh.nn), self.fixed)
        self.Vbc = np.zeros(mesh.nn)
        self.Vbc[self.D1] = 1.0
        self.Vbc[self.D2] = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.C = build_coupling(self.mesh, self.roi, Z_WIRING)
        self.Bzobj = GY * self.roi[:, 2]


def make_problem(axis: str, args):
    """Create the paper's axis-specific one-octant cylindrical problem."""
    classes = {"x": PhaseAlignedXProblem, "y": YProblem, "z": ZProblem}
    axis = axis.lower()
    if axis not in classes:
        raise ValueError("axis must be x, y, or z")
    return classes[axis](
        nl=int(args.nl),
        nz=int(args.nz),
        n_roi=int(getattr(args, "n_roi", 13)),
        p=5.0,
        alpha=float(args.alpha),
        volfrac=float(args.volume_fraction),
        rmin=float(getattr(args, "filter_radius", 1.0)),
    )


def initialize(problem, density) -> None:
    """Set the objective normalization using a reference density."""
    problem.init_norm = None
    _, _, info = problem.evaluate(np.asarray(density, dtype=float).ravel())
    problem.init_norm = (info["phi"], info["R"])

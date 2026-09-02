"""Biot-Savart coupling with 8-fold mirror symmetry + ROI sampling.

Bz_i = sum_e C[i,e] * Jl_e ,  where Jl_e is the signed circumferential
surface-current density (A/m) in base octant element e.

The full coil is the base octant replicated by the mirror group
{diag(sx,sy,sz): sx,sy,sz in +-1}.  Each copy's current direction gets a
wiring sign so the assembled field is a transverse (y) gradient.
"""
import numpy as np
from gradient_topopt import MU0, R0, R_ROI


def make_roi_points(n_per_axis=13):
    """Points inside the ROI sphere of radius R_ROI (paper ~997 pts)."""
    g = np.linspace(-R_ROI, R_ROI, n_per_axis)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    P = P[np.linalg.norm(P, axis=1) <= R_ROI + 1e-12]
    return P


def build_coupling(mesh, roi, wiring=(1, 1, 1), theta_offset=0.0):
    """C[m, ne] mapping base circumferential current Jl_e -> Bz at ROI pts."""
    wx, wy, wz = wiring
    # ``theta_offset`` rotates the one-octant design domain and its electrical
    # ports together.  The default preserves all existing calculations.
    theta0 = mesh.el_l / R0 + float(theta_offset)       # (ne,)
    x1 = R0*np.cos(theta0); y1 = R0*np.sin(theta0); z1 = mesh.el_z
    # base circumferential unit direction
    ux = -np.sin(theta0); uy = np.cos(theta0)  # (ne,)
    Ae = mesh.Ae
    m = roi.shape[0]; ne = mesh.ne
    C = np.zeros((m, ne))
    signs = [(sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
    for (sx, sy, sz) in signs:
        wsign = (wx if sx < 0 else 1)*(wy if sy < 0 else 1)*(wz if sz < 0 else 1)
        px = sx*x1; py = sy*y1; pz = sz*z1                 # (ne,)
        jx = wsign*sx*ux; jy = wsign*sy*uy                 # copy unit current dir
        # R = r_field - r_source : shape (m, ne)
        Rx = roi[:, 0][:, None] - px[None, :]
        Ry = roi[:, 1][:, None] - py[None, :]
        Rz = roi[:, 2][:, None] - pz[None, :]
        Rn = (Rx**2 + Ry**2 + Rz**2)**1.5
        # (J x R)_z = jx*Ry - jy*Rx   (jz=0)
        C += (jx[None, :]*Ry - jy[None, :]*Rx)/Rn
    C *= (MU0/(4*np.pi))*Ae
    return C


if __name__ == "__main__":
    from gradient_topopt import Mesh, element_matrices, T, SIGMA_CU, GY
    import scipy.sparse as sp, scipy.sparse.linalg as spla

    mesh = Mesh(48, 30)
    roi = make_roi_points(13)
    print("ROI points:", roi.shape[0])

    # --- quick FEM solve with uniform solid conductor to get a trial current ---
    Ke, Bl = element_matrices(mesh.dl, mesh.dz)
    nn = mesh.nn
    rows, cols, vals = [], [], []
    for e in range(mesh.ne):
        nd = mesh.elem[e]
        ke = T*SIGMA_CU*Ke
        for a in range(4):
            for b in range(4):
                rows.append(nd[a]); cols.append(nd[b]); vals.append(ke[a, b])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(nn, nn))

    # BC: y-gradient (Fig 3b): D1 = short segment top-left, D2 = right edge
    top = np.array([mesh.node_index(i, mesh.nz) for i in range(mesh.nl+1)])
    seg = max(1, mesh.nl//8)
    D1 = top[:seg+1]                      # top-left short segment
    right = np.array([mesh.node_index(mesh.nl, j) for j in range(mesh.nz+1)])
    D2 = right
    fixed = np.unique(np.concatenate([D1, D2]))
    val = np.zeros(nn); val[D1] = 1.0; val[D2] = 0.0
    free = np.setdiff1d(np.arange(nn), fixed)
    Kff = K[free][:, free]; Kfx = K[free][:, fixed]
    Vf = spla.spsolve(Kff, -Kfx @ val[fixed])
    V = val.copy(); V[free] = Vf

    # element circumferential current Jl = t*sigma*dV/dl
    Jl = np.zeros(mesh.ne)
    for e in range(mesh.ne):
        nd = mesh.elem[e]
        grad = Bl @ V[nd]        # [dV/dl, dV/dz]
        Jl[e] = T*SIGMA_CU*grad[0]

    # test all wiring sign combos, check which gives clean y-gradient
    x, y, z = roi[:, 0], roi[:, 1], roi[:, 2]
    best = None
    for wiring in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
        C = build_coupling(mesh, roi, wiring)
        Bz = C @ Jl
        # correlation with x, y, z, const
        def corr(u):
            return abs(np.corrcoef(Bz, u)[0, 1])
        cy, cx, cz = corr(y), corr(x), corr(z)
        print(f"wiring {wiring}:  corr_y={cy:.3f} corr_x={cx:.3f} corr_z={cz:.3f}")

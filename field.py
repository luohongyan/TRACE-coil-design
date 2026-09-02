"""Magnetic field B_z at arbitrary points from a converged octant current
distribution (8-fold mirror symmetry), field-inaccuracy metric, and
magnetic-energy / inductance."""
import numpy as np
from gradient_topopt import MU0, R0, R_ROI, GY, element_matrices

WIRING = (-1, 1, 1)


def field_at(mesh, pts, Jl_scaled):
    wx, wy, wz = WIRING
    theta0 = mesh.el_l/R0
    x1 = R0*np.cos(theta0); y1 = R0*np.sin(theta0); z1 = mesh.el_z
    ux = -np.sin(theta0); uy = np.cos(theta0)
    Ae = mesh.Ae
    Bz = np.zeros(pts.shape[0])
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                ws = (wx if sx < 0 else 1)*(wy if sy < 0 else 1)*(wz if sz < 0 else 1)
                px = sx*x1; py = sy*y1; pz = sz*z1
                jx = ws*sx*ux*Jl_scaled; jy = ws*sy*uy*Jl_scaled
                Rx = pts[:, 0][:, None]-px[None, :]
                Ry = pts[:, 1][:, None]-py[None, :]
                Rz = pts[:, 2][:, None]-pz[None, :]
                Rn = (Rx**2+Ry**2+Rz**2)**1.5
                Bz += ((jx[None, :]*Ry - jy[None, :]*Rx)/Rn).sum(axis=1)
    return Bz*(MU0/(4*np.pi))*Ae


def inaccuracy_metric(mesh, Jl_scaled, ngrid=17, rfrac=1.0):
    rr = rfrac*R_ROI
    g = np.linspace(-rr, rr, ngrid)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    P = P[np.linalg.norm(P, axis=1) <= rr]
    dy = 1e-4
    Bp = field_at(mesh, P+np.array([0, dy, 0]), Jl_scaled)
    Bm = field_at(mesh, P-np.array([0, dy, 0]), Jl_scaled)
    dBdy = (Bp-Bm)/(2*dy)
    dev = np.abs(GY-dBdy)/GY
    return dev.max(), dev, P


def magnetic_energy(mesh, sigma_e, V, Vin, T):
    if not hasattr(mesh, "_emat"):
        mesh._emat = element_matrices(mesh.dl, mesh.dz)
    _, Bl = mesh._emat
    Vnd = V[mesh.elem]
    gl = Vnd @ Bl[0]; gz = Vnd @ Bl[1]
    Jl = -sigma_e*gl*Vin
    Jz = -sigma_e*gz*Vin
    theta0 = mesh.el_l/R0
    ux = -np.sin(theta0); uy = np.cos(theta0)
    x1 = R0*np.cos(theta0); y1 = R0*np.sin(theta0); z1 = mesh.el_z
    dVol = mesh.Ae*T
    P = []; J = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                ws = (-1 if sx < 0 else 1)
                P.append(np.column_stack([sx*x1, sy*y1, sz*z1]))
                J.append(np.column_stack([ws*sx*ux*Jl, ws*sy*uy*Jl, ws*sz*Jz]))
    P = np.vstack(P); J = np.vstack(J)
    n = P.shape[0]
    reg = np.sqrt(mesh.Ae/np.pi)
    W = 0.0
    for a in range(0, n, 400):
        Pa = P[a:a+400]; Ja = J[a:a+400]
        d = np.sqrt(((Pa[:, None, :]-P[None, :, :])**2).sum(-1) + reg**2)
        W += ((Ja @ J.T)/d).sum()
    W *= (MU0/(8*np.pi))*dVol*dVol
    return W, None

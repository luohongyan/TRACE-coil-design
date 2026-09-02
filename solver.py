"""Two-objective SIMP topology optimization solver for the y-gradient coil.

Objective   f = fB + alpha*fR
  fB = phi/phi_init          magnetic-field linearity (least squares residual)
  fR = R/R_init              resistance (auxiliary, drives binary result)

FEM current continuity, Biot-Savart with 8-fold symmetry (wiring (-1,1,1)),
analytic optimal input voltage, adjoint sensitivities, density filter, OC update.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from gradient_topopt import (Mesh, element_matrices, MU0, SIGMA_CU, SIGMA_AIR,
                             T, GY, R0)
from biotsavart import make_roi_points, build_coupling

WIRING = (-1, 1, 1)   # verified: pure y-gradient


class Problem:
    def __init__(self, nl=64, nz=40, n_roi=13, p=5.0, alpha=100.0,
                 volfrac=0.1, rmin=1.6, d1_frac=0.125):
        self.mesh = Mesh(nl, nz)
        self.p = p
        self.alpha = alpha
        self.volfrac = volfrac
        self.roi = make_roi_points(n_roi)
        self.C = build_coupling(self.mesh, self.roi, WIRING)     # (m, ne)
        self.Bzobj = GY * self.roi[:, 1]                          # target = Gy*y
        self.Ke, self.Bl = element_matrices(self.mesh.dl, self.mesh.dz)
        self.bl_l = self.Bl[0]                                    # dV/dl row (4,)
        self._setup_bc(d1_frac)
        self._precompute_assembly()
        self._build_filter(rmin)
        self.init_norm = None       # (phi_init, R_init) set on first eval

    # ---- boundary conditions (Fig 3b) -----------------------------------
    def _setup_bc(self, d1_frac):
        m = self.mesh
        top = np.array([m.node_index(i, m.nz) for i in range(m.nl+1)])
        seg = max(1, int(round(m.nl*d1_frac)))
        self.D1 = top[:seg+1]                                    # top-left segment
        self.D2 = np.array([m.node_index(m.nl, j) for j in range(m.nz+1)])
        self.fixed = np.unique(np.concatenate([self.D1, self.D2]))
        self.free = np.setdiff1d(np.arange(m.nn), self.fixed)
        self.Vbc = np.zeros(m.nn)
        self.Vbc[self.D1] = 1.0
        self.Vbc[self.D2] = 0.0

    # ---- element -> global assembly index arrays ------------------------
    def _precompute_assembly(self):
        m = self.mesh
        nd = m.elem                                    # (ne,4)
        self.iK = np.kron(nd, np.ones((1, 4), int)).ravel()
        self.jK = np.kron(nd, np.ones((4, 1), int)).ravel()
        self.KeFlat = self.Ke.ravel()

    def _build_filter(self, rmin):
        m = self.mesh
        cl = m.el_l.reshape(m.nl, m.nz)
        cz = m.el_z.reshape(m.nl, m.nz)
        # neighbour weights H (linear hat), operate in element-index space
        R = int(np.ceil(rmin))
        rows, cols, vals = [], [], []
        idx = np.arange(m.ne).reshape(m.nl, m.nz)
        for i in range(m.nl):
            for j in range(m.nz):
                e = idx[i, j]
                for di in range(-R, R+1):
                    for dj in range(-R, R+1):
                        ii, jj = i+di, j+dj
                        if 0 <= ii < m.nl and 0 <= jj < m.nz:
                            f = idx[ii, jj]
                            d = np.hypot(di, dj)
                            w = rmin - d
                            if w > 0:
                                rows.append(e); cols.append(f); vals.append(w)
        self.Hf = sp.csr_matrix((vals, (rows, cols)), shape=(m.ne, m.ne))
        self.Hs = np.array(self.Hf.sum(axis=1)).ravel()

    def filt(self, x):
        return (self.Hf @ x)/self.Hs

    def filt_sens(self, dc):
        return self.Hf @ (dc/self.Hs)

    # ---- SIMP conductivity ---------------------------------------------
    def sigma(self, rho):
        return SIGMA_AIR + (SIGMA_CU-SIGMA_AIR)*rho**self.p

    def dsigma(self, rho):
        return self.p*(SIGMA_CU-SIGMA_AIR)*rho**(self.p-1)

    # ---- assemble & solve FEM ------------------------------------------
    def solve_fem(self, sig):
        m = self.mesh
        sK = (self.KeFlat[None, :]*(T*sig)[:, None]).ravel()
        K = sp.csr_matrix((sK, (self.iK, self.jK)), shape=(m.nn, m.nn))
        Kff = K[self.free][:, self.free]
        Kfx = K[self.free][:, self.fixed]
        rhs = -Kfx @ self.Vbc[self.fixed]
        Vf = spla.spsolve(Kff.tocsc(), rhs)
        V = self.Vbc.copy(); V[self.free] = Vf
        return K, V

    # ---- full evaluation: objective + adjoint gradient ------------------
    def evaluate(self, rho, want_grad=True):
        m = self.mesh
        sig = self.sigma(rho)
        dsig = self.dsigma(rho)
        K, V = self.solve_fem(sig)

        Vnd = V[m.elem]                       # (ne,4)
        g = Vnd @ self.bl_l                   # dV/dl per element (ne,)
        Jl = T*sig*g                          # circumferential current (ne,)
        Bz = self.C @ Jl                      # (m,)

        # optimal input voltage (eq A1)
        SBB = np.dot(Bz, self.Bzobj)
        SBz = np.dot(Bz, Bz)
        Vin = SBB/SBz if SBz > 0 else 0.0
        phi = 0.5*(np.dot(self.Bzobj, self.Bzobj) - SBB**2/SBz)

        Q = V @ (K @ V)                       # dissipated power
        R = 1.0/Q

        if self.init_norm is None:
            self.init_norm = (phi, R)
        phi0, R0n = self.init_norm
        fB = phi/phi0
        fR = R/R0n
        f = fB + self.alpha*fR

        info = dict(f=f, fB=fB, fR=fR, phi=phi, R=R, Vin=Vin, Bz=Bz, V=V,
                    Jl=Jl, Q=Q, current=Vin/R, Bscaled=Vin*Bz)
        if not want_grad:
            return f, None, info

        # ---------- adjoint sensitivity ----------
        resid = (Vin*Bz - self.Bzobj)         # (m,)
        # dBz/dV : a_i vector (nn,). Bz_i = sum_e C_ie T sig_e (bl . Vnd_e)
        # dphi/dV = sum_i resid_i * Vin * dBz_i/dV
        w = resid*Vin                         # (m,)
        # coefficient per element: cw_e = sum_i w_i C_ie * T sig_e
        cw = (w @ self.C)*(T*sig)             # (ne,)
        dphidV = np.zeros(m.nn)
        contrib = cw[:, None]*self.bl_l[None, :]     # (ne,4)
        np.add.at(dphidV, m.elem, contrib)
        dfBdV = dphidV/phi0

        KV = K @ V
        dfRdV = (-2.0*KV/Q**2)/R0n            # d(fR)/dV

        dfdV = dfBdV + self.alpha*dfRdV

        # adjoint solve K lam = -dfdV (free block)
        lam = np.zeros(m.nn)
        Kff = K[self.free][:, self.free].tocsc()
        lam[self.free] = spla.spsolve(Kff, -dfdV[self.free])

        # explicit partials wrt rho
        # dBz_i/drho_e (explicit) = C_ie T dsig_e g_e
        dBz_drho_expl = (self.C * (T*dsig*g)[None, :])   # (m, ne)
        dphi_drho = (w @ dBz_drho_expl)                  # (ne,)
        dfB_drho = dphi_drho/phi0

        # dQ/drho_e (explicit) = T dsig_e (Vnd_e^T Ke Vnd_e)
        VKeV = np.einsum('ea,ab,eb->e', Vnd, self.Ke, Vnd)
        dQ_drho = T*dsig*VKeV
        dfR_drho = (-1.0/Q**2)*dQ_drho/R0n

        df_drho_expl = dfB_drho + self.alpha*dfR_drho

        # adjoint term: lam^T (dK/drho_e) V = T dsig_e (lam_e^T Ke Vnd_e)
        Lnd = lam[m.elem]
        lamKeV = np.einsum('ea,ab,eb->e', Lnd, self.Ke, Vnd)
        df_adj = T*dsig*lamKeV

        df_drho = df_drho_expl + df_adj
        return f, df_drho, info

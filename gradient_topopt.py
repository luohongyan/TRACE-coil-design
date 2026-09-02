"""
Reproduction of:
Hui Pan et al., "Design of small-scale gradient coils in MRI by using the
topology optimization method", Chin. Phys. B 27(5), 050201 (2018).

y-gradient coil, two-objective SIMP topology optimization.

Method
------
Design domain = one octant (Omega_1) of the developed (unrolled) cylindrical
current-carrying surface.  Design variable rho = distribution of conductive
material.  Conductivity by SIMP:  sigma(rho) = sigma_air + (sigma_cu-sigma_air) rho^p.

Steps per iteration (paper Sec 2.3, Fig 4):
  1. Solve current-continuity FEM  K(rho) V = P  (unit voltage drive, Fig 3b).
  2. Circumferential surface current  Jl = t*sigma*dV/dl.
  3. Biot-Savart -> Bz at ROI points, replicated to the full cylinder by the
     x=0,y=0,z=0 mirror symmetry of a transverse (y) gradient coil.
  4. Analytic optimal input voltage Vin (eq A1); objective f = fB + alpha*fR.
  5. Adjoint sensitivities df/drho (Appendix A).
  6. OC update with volume constraint; density filtering.

Author: reproduction script.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------------
# Physical constants / paper parameters
# ----------------------------------------------------------------------------
MU0        = 4e-7 * np.pi
SIGMA_CU   = 5.99e7          # S/m
SIGMA_AIR  = 5.0e-15         # S/m
R0         = 10e-3           # radius of current-carrying surface (m)
H          = 20e-3           # axial height (m)
T          = 1e-3            # surface (conductor) thickness (m)
R_ROI      = 5e-3            # ROI radius (m)
GY         = 0.01            # target y-gradient (T/m)

L0         = 2*np.pi*R0      # full circumference (developed width)
W_OCT      = L0/4.0          # octant width  (l in [0, L0/4])
H_OCT      = H/2.0           # octant height (z in [0, H/2])


# ----------------------------------------------------------------------------
# Mesh of the octant Omega_1  (bilinear quad elements)
# ----------------------------------------------------------------------------
class Mesh:
    def __init__(self, nl, nz):
        self.nl, self.nz = nl, nz
        self.ne = nl*nz
        self.nn = (nl+1)*(nz+1)
        # nodal coordinates (l,z)
        ls = np.linspace(0, W_OCT, nl+1)
        zs = np.linspace(0, H_OCT, nz+1)
        LL, ZZ = np.meshgrid(ls, zs, indexing='ij')   # shape (nl+1, nz+1)
        self.node_l = LL.ravel()
        self.node_z = ZZ.ravel()
        self.dl = W_OCT/nl
        self.dz = H_OCT/nz
        self.Ae = self.dl*self.dz
        # element -> node connectivity (4 nodes, CCW)
        nid = np.arange(self.nn).reshape(nl+1, nz+1)
        el = np.zeros((self.ne, 4), dtype=int)
        e = 0
        for i in range(nl):
            for j in range(nz):
                el[e] = [nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1]]
                e += 1
        self.elem = el
        # element centre (l,z)
        self.el_l = self.node_l[el].mean(axis=1)
        self.el_z = self.node_z[el].mean(axis=1)

    def node_index(self, i, j):
        return i*(self.nz+1)+j


# ----------------------------------------------------------------------------
# Bilinear element matrices on a rectangle dl x dz
# ----------------------------------------------------------------------------
def element_matrices(dl, dz):
    """Return Ke (4x4 stiffness for unit conductivity) and Bl (2x4 -> dV/dl,
    dV/dz gradient operator evaluated at element centre)."""
    # 2x2 Gauss
    g = 1.0/np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
    Ke = np.zeros((4, 4))
    for (xi, eta) in gp:
        dNdxi  = 0.25*np.array([-(1-eta),  (1-eta), (1+eta), -(1+eta)])
        dNdeta = 0.25*np.array([-(1-xi), -(1+xi),  (1+xi),   (1-xi)])
        # jacobian (rectangle): dl/2, dz/2
        dNdl = dNdxi*(2.0/dl)
        dNdz = dNdeta*(2.0/dz)
        B = np.vstack([dNdl, dNdz])
        detJ = (dl/2)*(dz/2)
        Ke += (B.T @ B)*detJ
    # gradient at centre (xi=eta=0)
    dNdxi  = 0.25*np.array([-1,  1, 1, -1])
    dNdeta = 0.25*np.array([-1, -1, 1,  1])
    Bl = np.vstack([dNdxi*(2.0/dl), dNdeta*(2.0/dz)])   # rows: d/dl, d/dz
    return Ke, Bl


if __name__ == "__main__":
    m = Mesh(8, 5)
    Ke, Bl = element_matrices(m.dl, m.dz)
    print("mesh ne,nn:", m.ne, m.nn)
    print("Ke row sums (should be ~0):", np.round(Ke.sum(axis=1), 12))
    print("Bl:\n", Bl)
    print("W_OCT(mm)=", W_OCT*1e3, "H_OCT(mm)=", H_OCT*1e3, "L0(mm)=", L0*1e3)

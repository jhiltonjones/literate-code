import numpy as np 
import osqp
from scipy.linalg import solve_discrete_are, block_diag
import scipy.sparse as sp

def dare_stabilising_K(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P

def trust_radius(jac_fn, psi_rad, *,
                 h_rad=np.deg2rad(0.5),
                 eps_theta_rad=np.deg2rad(1.0),
                 Jmin=1e-6, Lmin=1e-6,
                 dpsi_cap=np.deg2rad(50.0)):
    J0 = float(jac_fn(psi_rad))
    Jp = float(jac_fn(psi_rad + h_rad))
    Jm = float(jac_fn(psi_rad - h_rad))
    Jprime = (Jp - Jm) / (2.0*h_rad)
    dpsi_lin  = eps_theta_rad / max(abs(J0),    Jmin)
    dpsi_quad = (2.0*eps_theta_rad / max(abs(Jprime), Lmin))**0.5
    dpsi = min(dpsi_lin, dpsi_quad, dpsi_cap)
    return dpsi

def seq_mat_tv(Phi_list, B_list):
    N = len(Phi_list)
    n,m = B_list[0].shape   
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))
    P = np.eye(n)

    for i in range(N):
        P = Phi_list[i] @ P
        Mx[i*n:(i+1)*n, :] = P
    
    Mc[0:n, 0:m] = B_list[0]

    for i in range(1,N):
        Mc[i*n:(i+1)*n, 0:i*m] = Phi_list[i] @ Mc[(i-1)*n:i*n, 0:i*m]
        Mc[i*n:(i+1)*n, i*m:(i+1)*m] = B_list[i]

    return Mx, Mc
    
def seq_mat_lti(A, B, N):

    n, m = B.shape
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))

    A_pow = np.eye(n)

    A_pow = A @ A_pow          
    Mx[0:n, :] = A_pow
    Mc[0:n, 0:m] = B
    for i in range(1, N):
        A_pow = A @ A_pow  
        Mx[i*n:(i+1)*n, :] = A_pow
        Mc[i*n:(i+1)*n, 0:i*m] = A @ Mc[(i-1)*n:i*n, 0:i*m]
        Mc[i*n:(i+1)*n, i*m:(i+1)*m] = B

    return Mx, Mc


def solve_qp_osqp(H,f,A, l, u, U_warm = None):
    P = sp.csc_matrix(0.5 * (H+H.T)) #This is a sparse matrix which only keeps non-zero entries for efficiency 
    q = f.astype(float)#Expects this as a float 64 and will fail if this is not the case
    A = sp.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u)
    if U_warm is not None:
        prob.warm_start(x=U_warm)
    res = prob.solve()
    status = res.info.status
    if status not in ("solved", "solved inaccurate"):
        return None, None, status
    return res.x, res.y, status

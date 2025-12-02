import numpy as np

from numpy.polynomial.legendre import leggauss


def build_k(E, r, nu, lengths):
    E  = np.atleast_1d(E); nu = np.atleast_1d(nu)
    L  = np.atleast_1d(lengths); r = np.atleast_1d(r)
    if r.size == 1: r = np.full(len(L), r.item())
    K = np.zeros((3*len(L), 3*len(L)))
    for i in range(len(L)):
        I_i = (np.pi*r[i]**4)/4      
        J_i = 2.0 * I_i           
        G_i = E[i] / (2.0 * (1.0 + nu[i]))
        Ki  = np.diag([E[i]*I_i, E[i]*I_i, G_i*J_i])
        K[3*i:3*i+3, 3*i:3*i+3] = Ki
    return K


def G_matrix_build(axes, thetas, lengths, masses, R_base = None, p_base=None, quad_n=12):
    """
    This builds the G matrix in equation 16
    This is computed by M which is the mass density* matrix of stacked jacobians * g
    """

    n = len(lengths)
    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()

    S = np.zeros((3*n, 3))

    for i, (n_hat, theta, L) in enumerate(zip(axes, thetas, lengths)):
        gamma_i = (theta/L)*np.asarray(n_hat)
        xi, wi = leggauss(quad_n)
        si = 0.5*(xi+1)*L
        w = 0.5 * L*wi

        Int = np.zeros((3,3))

        for s_k, w_k in zip(si,w):
            Jp,_ = local_jacobians_from_gamma(gamma_i, s_k, quad_n=max(6,quad_n//2))
            Jp = R_up @ Jp
            Int += w_k * Jp.T
        S[3*i:3*i+3, :] = Int

        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_i, L)

    M = np.zeros((3*n, 3*n))
    for i, m in enumerate(masses):
        M[3*i: 3*i+3, 3*i:3*i+3] = m * np.eye(3)
    def apply_to_g(g):
        g = np.asarray(g).reshape(3)
        return M@(S@g)  
    return apply_to_g, S 
def energy_and_grad(axes, thetas, lengths, masses, gvec, E, r, nu,
                    R_base=None, p_base=None, quad_n=12):
    """
    This codes equation 13 and equation 15. The U grad codes the second part in equation 13
    """
    gamma = np.concatenate([(theta/L)*np.asarray(n_hat)
                            for n_hat, theta, L in zip(axes, thetas, lengths)])

    K = build_k(np.asarray(E), np.asarray(r), np.asarray(nu), np.asarray(lengths))
    U_stiff = 0.5 * gamma @ (K @ gamma)

    U_grav = 0.0
    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()
    for n_hat, theta, L, m in zip(axes, thetas, lengths, masses):
        gamma_i = (theta/L) * np.asarray(n_hat,float)
        xi, wi = leggauss(quad_n)
        si = 0.5*(xi+1.0)*L
        w  = 0.5*L*wi
        for s, ws in zip(si, w):
            _, p_s = pose_from_gamma(R_up, p_up, gamma_i, s)
            U_grav += - m * (gvec @ p_s) * ws
        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_i, L)

    apply_G, _S = G_matrix_build(axes, thetas, lengths, masses, R_base, p_base, quad_n)
    Qg = apply_G(gvec)     

    grad = (K @ gamma) - Qg

    U = U_stiff + U_grav
    return U, grad, K

def H_matrix_build(axes, thetas, lengths, eta0_list,
                   R_base=None, p_base=None, quad_n=12):
    """
    Build H(γ) so that τ = H(γ) B + J(γ)^T f_e  (Eq. 19).

    eta0_list: list of 3-vectors η_{j0} (magnetic dipole per unit length) in each segment's local frame.
    """
    n = len(lengths)
    H = np.zeros((3*n, 3))

    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()

    for j, (n_hat, theta, L, eta0) in enumerate(zip(axes, thetas, lengths, eta0_list)):
        gamma_j = (theta / L) * np.asarray(n_hat, dtype=float)

        xi, wi = leggauss(quad_n)
        si = 0.5 * (xi + 1.0) * L
        ws = 0.5 * L * wi

        Hj = np.zeros((3, 3))
        for s_k, w_k in zip(si, ws):
            R_s, _ = pose_from_gamma(R_up, p_up, gamma_j, s_k)
            _, Jo_loc = local_jacobians_from_gamma(gamma_j, s_k, quad_n=max(6, quad_n//2))
            Jo_world = R_s @ Jo_loc   
            eta_s = R_s @ np.asarray(eta0)
            Hj += w_k * (Jo_world.T @ hat(eta_s))
        H[3*j:3*j+3, :] = Hj
        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_j, L)

    return H

def tau_from_eq19(axes, thetas, lengths, eta0_list, B, f_e,
                  R_base=None, p_base=None, quad_n=12):
    """
    Compute τ = H(γ) B + J(γ)^T f_e   exactly as Eq. (19).

    - B: 3-vector external magnetic field in world frame.
    - f_e: stacked 6n-vector of concentrated wrenches at each segment end
           [f1; m1; f2; m2; ...] in world frame.
    """
    H = H_matrix_build(axes, thetas, lengths, eta0_list, R_base, p_base, quad_n)
    _, _, J, _, _ = stack_blocks(axes, thetas, lengths, s_list=lengths,
                                 R_base=R_base, p_base=p_base)

    tau = H @ np.asarray(B).reshape(3) + J.T @ np.asarray(f_e).reshape(-1)
    return tau, H, J

def eq20_residual(axes, thetas, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
                  R_base=None, p_base=None, quad_n=12):
    """
    Residual of Eq. (20): r = [-Kγ - G(γ)] - [H(γ)B + J(γ)^T f_e]
    Returns (r, pieces...) so you can debug/inspect.
    """
    gamma = np.concatenate([(theta/L)*np.asarray(n_hat, float)
                            for n_hat, theta, L in zip(axes, thetas, lengths)])
    K = build_k(np.asarray(E), np.asarray(r), np.asarray(nu), np.asarray(lengths))
    Kgamma = K @ gamma
    apply_G, _S = G_matrix_build(axes, thetas, lengths, masses,
                                 R_base=R_base, p_base=p_base, quad_n=quad_n)
    Qg = apply_G(gvec)  
    H = H_matrix_build(axes, thetas, lengths, eta0_list,
                       R_base=R_base, p_base=p_base, quad_n=quad_n)
    _, _, J, _, _ = stack_blocks(axes, thetas, lengths, s_list=lengths,
                                 R_base=R_base, p_base=p_base)

    right = H @ np.asarray(B).reshape(3) + J.T @ np.asarray(f_e).reshape(-1)
    left  = -Kgamma - Qg
    r = left - right
    return r, left, right, K, H, J, Qg, gamma


def central_difference_jacobian(func, x, rel_eps=1e-6, abs_eps=1e-10, h_floor=1e-4, *args, **kwargs):
    x = np.asarray(x, float)
    n = x.size
    r0 = np.asarray(func(x, *args, **kwargs), float)
    m = r0.size
    J = np.zeros((m, n))
    for j in range(n):
        h = max(abs_eps, rel_eps * max(1.0, abs(x[j])), h_floor)
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        rp = np.asarray(func(xp, *args, **kwargs), float)
        rm = np.asarray(func(xm, *args, **kwargs), float)
        J[:, j] = (rp - rm) / (2.0 * h)
    return r0, J

def residual_theta(thetas, axes, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
                   R_base=None, p_base=None, quad_n=12):
    r, *_ = eq20_residual(axes, thetas, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
                          R_base=R_base, p_base=p_base, quad_n=quad_n)
    return r
# def guass_newton_lm(thetas0, axes, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
#                     R_base=None, p_base=None, quad_n=12,
#                     max_iter=50, tol_r=1e-8, tol_step=1e-8,
#                     lambda_init=1e-8, lambda_up=10.0, lambda_down=0.1,
#                     h_floor=5e-3):
#     theta = np.asarray(thetas0, float).copy()
#     lam = float(lambda_init)

#     for k in range(max_iter):
#         res, J = central_difference_jacobian(
#             residual_theta, theta,
#             rel_eps=1e-6, abs_eps=1e-10, h_floor=h_floor,
#             axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
#             B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
#             R_base=R_base, p_base=p_base, quad_n=quad_n
#         )
#         res_norm = np.linalg.norm(res)
#         if res_norm < tol_r:
#             break

#         JTJ  = J.T @ J
#         grad = J.T @ res                         
#         jtj_scale = float(np.max(np.diag(JTJ)))
#         jtj_scale = max(jtj_scale, 1e-20)
#         A_sys = JTJ + (lam * jtj_scale) * np.eye(JTJ.shape[0])

#         try:
#             step = -np.linalg.solve(A_sys, grad)
#         except np.linalg.LinAlgError:
#             step = -np.linalg.pinv(A_sys) @ grad

#         if np.linalg.norm(step, np.inf) < tol_step:
#             break

#         max_step = 0.5
#         step = np.clip(step, -max_step, max_step)

#         t = 1.0
#         while t > 1e-6:
#             res_try = residual_theta(theta + t*step, axes, lengths, masses, eta0_list,
#                                      B, f_e, gvec, E, r, nu,   # <- pass gvec and r unchanged
#                                      R_base=R_base, p_base=p_base, quad_n=quad_n)
#             if np.linalg.norm(res_try) < res_norm:
#                 theta = theta + t*step
#                 lam   = max(lam * lambda_down, 1e-12)
#                 break
#             t *= 0.5

#         if t <= 1e-6:
#             lam *= lambda_up

#     return theta
def gauss_newton_lm(thetas0, axes, lengths, masses, eta0_list, B, f_e,
                          gvec, E, r, nu,
                          R_base=None, p_base=None, quad_n=12,
                          max_iter=50,
                          tol_grad=1e-8,      # NEW: gradient stopping
                          tol_step=1e-10,     # relative step stopping
                          tol_r=1e-8,
                          lambda_init=1e-3,   # typical LM starting value
                          h_floor=5e-3):
    """
    Full Gavin-style Levenberg-Marquardt with rho update and no line search.
    """

    theta = np.asarray(thetas0, float).copy()
    lam = float(lambda_init)

    for k in range(max_iter):

        # ---- Evaluate residual and Jacobian ---------
        res, J = central_difference_jacobian(
            residual_theta, theta,
            rel_eps=1e-6, abs_eps=1e-10, h_floor=h_floor,
            axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
            B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )

        res_norm = np.linalg.norm(res)
        if res_norm < tol_r:
            break

        # ---- First-order optimality test ----
        grad = J.T @ res
        if np.linalg.norm(grad, np.inf) < tol_grad:
            break

        # ---- Build LM system ----------
        JTJ = J.T @ J
        scale = float(np.max(np.diag(JTJ)))
        scale = max(scale, 1e-12)

        A = JTJ + lam * scale * np.eye(len(theta))

        try:
            h = -np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            h = -np.linalg.pinv(A) @ grad

        # ---- Relative step test ----
        if np.linalg.norm(h) < tol_step * (np.linalg.norm(theta) + 1e-12):
            break

        # ---- Compute predicted reduction (Gavin Eq. 15/16) ----
        # pred = | hᵀ ( λ h + Jᵀ r ) |
        pred = np.abs(h @ (lam * scale * h + grad))

        if pred < 1e-30:   # step too small to be meaningful
            break

        # ---- Evaluate new residual ----
        theta_new = theta + h
        res_new = residual_theta(theta_new, axes, lengths, masses, eta0_list,
                                 B, f_e, gvec, E, r, nu,
                                 R_base=R_base, p_base=p_base, quad_n=quad_n)

        res_new_norm = np.linalg.norm(res_new)

        # ---- Actual reduction ----
        act = 0.5*(res_norm**2 - res_new_norm**2)

        # ---- Compute rho (Eq. 14) ----
        rho = act / pred

        # ---- Update step depending on rho (Gavin Sec 4.1.1 method 3) ----
        if rho > 0:      # success → accept step
            theta = theta_new
            res = res_new
            res_norm = res_new_norm

            # decrease lambda (more Gauss-Newton-like)
            lam = lam * max(1/3, 1 - (2*rho - 1)**3)
            lam = max(lam, 1e-15)

        else:            # failure → reject step
            # increase lambda (more gradient-descent-like)
            lam = lam * min(10, (1/(1+rho)))
            continue

    return theta

def magnetic_field_for_theta(thetas, axes, lengths, masses, eta0_list,
                             f_e, gvec, E, r, nu,
                             R_base=None, p_base=None, quad_n=12,
                             prefer_dir=None,  # e.g. np.array([0,0,1]) to force B along z
                             reg=0.0):         # Tikhonov (ridge) regularization
    """
    Solve Eq. (20) for B given thetas (static equilibrium):
        [-Kγ - G(γ)] = H(γ) B + J(γ)^T f_e
      => H B = -Kγ - Qg - J^T f_e  (linear in B)

    If prefer_dir is provided (3-vector), solve only for magnitude along that direction.
    """
    thetas = np.asarray(thetas, float)
    # gamma from thetas
    gamma = np.concatenate([(theta/L)*np.asarray(n_hat, float)
                            for n_hat, theta, L in zip(axes, thetas, lengths)])

    # Elastic term
    K = build_k(np.asarray(E), np.asarray(r), np.asarray(nu), np.asarray(lengths))

    # Gravity generalized load Qg
    apply_G, _ = G_matrix_build(axes, thetas, lengths, masses,
                                R_base=R_base, p_base=p_base, quad_n=quad_n)
    Qg = apply_G(gvec)

    # Magnetic mapping H and Jacobian J for concentrated wrenches
    H = H_matrix_build(axes, thetas, lengths, eta0_list,
                       R_base=R_base, p_base=p_base, quad_n=quad_n)
    _, _, J, _, _ = stack_blocks(axes, thetas, lengths, s_list=lengths,
                                 R_base=R_base, p_base=p_base)

    # Right-hand side for H B = rhs
    rhs = -(K @ gamma) - Qg - (J.T @ np.asarray(f_e).reshape(-1))

    # Solve for B
    # if prefer_dir is not None:
    #     # Constrain B = alpha * u
    #     u = np.asarray(prefer_dir, float).reshape(3)
    #     nu = np.linalg.norm(u)
    #     if nu == 0:
    #         raise ValueError("prefer_dir must be nonzero")
    #     u = u / nu
    #     Hu = H @ u
    #     denom = (Hu @ Hu) + float(reg)
    #     if denom < 1e-16:
    #         raise np.linalg.LinAlgError("Direction leads to near-singular system.")
    #     alpha = (Hu @ rhs) / denom
    #     B = alpha * u
    # else:
        # Unconstrained least squares (with optional ridge)
        # Solve (H^T H + reg I) B = H^T rhs
    HtH = H.T @ H
    if reg > 0:
        HtH = HtH + reg * np.eye(3)
    try:
        B = np.linalg.solve(HtH, H.T @ rhs)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(H) @ rhs  # fallback

    # Diagnostics
    res = H @ B - rhs
    info = {
        "rhs_norm": float(np.linalg.norm(rhs)),
        "residual_norm": float(np.linalg.norm(res)),
        "H_cond_est": float(np.linalg.cond(H)) if np.all(np.isfinite(H)) else np.inf,
        "B_T": B
    }
    return B, info
def my_gauss_newton_lm(thetas0, axes, lengths, masses, eta0_list, B, f_e,
                       gvec, E, r, nu,
                       R_base=None, p_base=None, quad_n=12,
                       max_iter=80,
                       tol_grad=1e-10,
                       tol_step=1e-12,
                       tol_r=1e-10,
                       lambda_init=1e-4,
                       h_floor=1e-5):
    theta = np.asarray(thetas0, float).copy()
    lam = float(lambda_init)

    for _ in range(max_iter):
        # residual and Jacobian of *residual_theta*, not eq20_residual
        res, J = central_difference_jacobian(
            residual_theta, theta,
            rel_eps=1e-6, abs_eps=1e-10, h_floor=h_floor,
            axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
            B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )
        res_norm = np.linalg.norm(res)
        if res_norm < tol_r:
            break

        grad = J.T @ res
        if np.linalg.norm(grad, np.inf) < tol_grad:
            break

        JTJ = J.T @ J
        scale = max(np.max(np.diag(JTJ)), 1e-12)
        A = JTJ + lam * scale * np.eye(len(theta))

        try:
            h = -np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            h = -np.linalg.pinv(A) @ grad

        if np.linalg.norm(h) < tol_step * (np.linalg.norm(theta) + 1e-12):
            break

        theta_new = theta + h
        # evaluate new residual
        res_new = residual_theta(
            theta_new, axes, lengths, masses, eta0_list,
            B, f_e, gvec, E, r, nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )
        res_new_norm = np.linalg.norm(res_new)

        act = 0.5 * (res_norm**2 - res_new_norm**2)
        pred = np.abs(h @ (lam * scale * h + grad))
        rho = act / (pred + 1e-30)

        if rho > 0:
            theta = theta_new
            res = res_new
            res_norm = res_new_norm
            lam = lam * max(1/3, 1 - (2*rho - 1)**3)
            lam = max(lam, 1e-15)
        else:
            lam = lam * min(10, 1/(1+rho))

    return theta


def hat(v):
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def so3_exp(omega):
    th=np.linalg.norm(omega); W=hat(omega)
    if th<1e-12: return np.eye(3)+W
    W2=W@W; s,c=np.sin(th),np.cos(th)
    return np.eye(3)+(s/th)*W+((1-c)/(th**2))*W2
def so3_left_jacobian(omega):
    th=np.linalg.norm(omega); W=hat(omega)
    if th<1e-12: return np.eye(3)+0.5*W+(1/6)*(W@W)
    W2=W@W; s,c=np.sin(th),np.cos(th)
    return np.eye(3)+((1-c)/(th**2))*W+((th-s)/(th**3))*W2
def so3_right_jacobian(omega):
    return so3_left_jacobian(-np.asarray(omega))
def pose_from_gamma(R0,p0,gamma,s):
    R=R0@so3_exp(gamma*s)
    e3=np.array([0.,1.,0.])
    p=p0+R0@(s*so3_left_jacobian(gamma*s)@e3)
    return R,p

def local_jacobians_from_gamma(gamma, s, quad_n=16):
    Jo_local = s * so3_right_jacobian(gamma * s)
    e3 = np.array([0.0, 1.0, 0.0])
    Jp_local = np.zeros((3, 3))
    xi, wi = leggauss(quad_n)
    ui = 0.5 * (xi + 1.0) * s
    w_scaled = 0.5 * s * wi
    for j in range(3):
        ej = np.eye(3)[:, j]
        col = np.zeros(3)
        for u, w in zip(ui, w_scaled):
            R_rel = so3_exp(gamma * u)
            Jr = so3_right_jacobian(gamma * u)
            v = Jr @ (u * ej)
            col += w * (R_rel @ (hat(v) @ e3))
        Jp_local[:, j] = col
    return Jp_local, Jo_local

def stack_blocks(axes, thetas, lengths, s_list=None, R_base=None, p_base=None):
    n = len(lengths)
    if s_list is None:
        s_list = lengths                          

    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()

    R_nodes = [R_up.copy()]
    p_nodes = [p_up.copy()]
    Rs_si, ps_si = [], []
    J = np.zeros((6*n, 3*n))

    for i, (n_hat, theta, L, s_i) in enumerate(zip(axes, thetas, lengths, s_list)):
        gamma_i = (theta / L) * np.asarray(n_hat) 

        Jp_loc, Jo_loc = local_jacobians_from_gamma(gamma_i, s_i)

        R_si, p_si = pose_from_gamma(R_up, p_up, gamma_i, s_i)
        Rs_si.append(R_si)
        ps_si.append(p_si)
        Jp_world = R_up @ Jp_loc
        Jo_world = R_si @ Jo_loc

        J_block = np.vstack([Jp_world, Jo_world])
        J[6*i:6*i+6, 3*i:3*i+3] = J_block

        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_i, L)
        R_nodes.append(R_up.copy())
        p_nodes.append(p_up.copy())

    return np.array(Rs_si), np.array(ps_si), J, np.array(R_nodes), np.array(p_nodes)

prefer_dir = None
# ======================
#  Geometry & Materials
# ======================

L_cat = 0.1        # total catheter length [m]
L1    = L_cat      # single-segment length [m]

r1 = 0.001         # catheter radius [m]

E_niti   = 3e6
rho_niti = 6700
nu_niti  = 0.33

# z is UP (+z) / DOWN (-z); gravity acts along -z
gvec     = np.array([0.0, 0.0, -9.8])   # <<< CHANGED

A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0
EI1  = E_niti * I1

lam    = rho_niti * A_cs
lengths = np.array([L1], float)
n       = len(lengths)
masses  = lam * lengths

# Bending axis:
#   Beam lies along +y, bending is rotation about +x → deflection in (y,z) plane.
axes = [np.array([0.0, 0.0, 1.0], float)]   # bend about local/world +z

r_arr  = np.array([r1],       dtype=float)
E_arr  = np.array([E_niti],   dtype=float)
nu_arr = np.array([0.49],     dtype=float)
f_e    = np.zeros(6 * n, float)

# ======================
#  Internal Magnet (IPM)
# ======================

Br   = 1.2
mu0  = 4e-7 * np.pi

D_ipm = 2e-3
L_ipm = 4e-2
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm

mu_ipm_scalar = Br * V_ipm / mu0

# Magnetization direction in local/base frame:
# still along +z in beam frame (this is internal; OK to keep)
m_dir_local = np.array([0.0, 1.0, 0.0], float)

eta_ipm   = (mu_ipm_scalar / L_ipm) * m_dir_local
eta0_list = [eta_ipm]

print("Linear dipole density eta_ipm [A·m]:", eta_ipm)

# ======================
#  Target bend & solve for required B
# ======================

theta_target_deg = 10.0
thetas_target    = np.array([np.deg2rad(theta_target_deg)])
thetas0          = np.array([np.deg2rad(5.0)])

B_req, info = magnetic_field_for_theta(
    thetas_target,
    axes, lengths, masses, eta0_list,
    f_e=f_e,
    gvec=gvec,          # <<< uses new gravity
    E=E_arr,
    r=r_arr,
    nu=nu_arr,
    R_base=None,
    p_base=None,
    quad_n=12,
    prefer_dir=None,
    reg=0.0
)

theta_total = np.sum(thetas_target)

print("\n=== Field needed for target bend ===")
print("Overall target (deg):", np.rad2deg(theta_total))
print("Per-segment thetas_target (deg):", np.rad2deg(thetas_target))
print("Required B (T):", B_req)
print("Required |B| (mT):", np.linalg.norm(B_req) * 1e3)
print("||H B - rhs||:", info["residual_norm"], "cond(H):", info["H_cond_est"])

# ======================
#  Check equilibrium
# ======================

r0, J0 = central_difference_jacobian(
    residual_theta,
    thetas0,
    axes=axes,
    lengths=lengths,
    masses=masses,
    eta0_list=eta0_list,
    B=B_req,
    f_e=f_e,
    gvec=gvec,          # <<< new gravity
    E=E_arr,
    r=r_arr,
    nu=nu_arr
)

print("\n=== Newton initial state diagnostics ===")
print("Initial guess thetas0 (deg):", np.rad2deg(thetas0))
print("||r(thetas0)|| =", np.linalg.norm(r0))
print("Column norms(J(thetas0)) =", np.linalg.norm(J0, axis=0))

thetas_sol = my_gauss_newton_lm(
    thetas0,
    axes, lengths, masses, eta0_list,
    B=B_req, f_e=f_e, gvec=gvec,      # <<< new gravity
    E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12,
    max_iter=80,
    tol_grad=1e-10,
    tol_step=1e-12,
    tol_r=1e-10,
    lambda_init=1e-4,
    h_floor=1e-5
)

print("\n=== Newton solution ===")
print("Solved thetas (deg):", np.rad2deg(thetas_sol))
print("Sum(theta) achieved (deg):", np.rad2deg(np.sum(thetas_sol)))
print("Error vs target (deg):", np.rad2deg(np.sum(thetas_sol) - theta_total))

# ======================
#  Helper: y,z -> thetas  (beam along +y, bending about +x)
# ======================
def thetas_from_xy(x, y, lengths, L_cat):
    """
    Map desired tip (x,y) to segment bending angles under
    a planar constant-curvature assumption in the x–y plane.

    Assumes:
      - beam base at origin,
      - initial beam axis along +y,
      - bending about +z (so deflection is in x–y).
    """

    y_safe = y if abs(y) > 1e-6 else 1e-6
    theta_tot = 2.0 * np.arctan2(x, y_safe)  # <<< note: atan2(x, y)

    lengths = np.asarray(lengths, float)
    thetas = theta_tot * (lengths / float(L_cat))
    return thetas
# def thetas_from_yz(y, z, lengths, L_cat):
#     """
#     Map desired tip (y,z) to segment bending angles under
#     a planar constant-curvature assumption in the y–z plane.

#     Assumes:
#       - beam base at origin,
#       - initial beam axis along +y,
#       - bending about +x (so deflection is in y–z).

#     For a constant-curvature arc (beam along +y):
#         y = R sin(theta_tot)
#         z = R (1 - cos(theta_tot))
#       => z / y = tan(theta_tot / 2)
#          theta_tot = 2 * atan2(z, y)
#     """
#     y_safe = y if abs(y) > 1e-6 else 1e-6
#     theta_tot = 2.0 * np.arctan2(z, y_safe)   # <<< CHANGED (was atan2(y, z))

#     lengths = np.asarray(lengths, float)
#     thetas = theta_tot * (lengths / float(L_cat))
#     return thetas

def tip_position_from_theta(axes, thetas, lengths,
                            R_base=None, p_base=None):
    Rs_si, ps_si, J, R_nodes, p_nodes = stack_blocks(
        axes, thetas, lengths,
        s_list=lengths,
        R_base=R_base, p_base=p_base
    )
    p_tip = p_nodes[-1]
    return p_tip

# ======================
#  Single EPM from B
# ======================
import numpy as np

def single_epm_pose_from_B_unrestricted(B_req, mu_mag, e_r, mu0=4e-7*np.pi):
    """
    Given required B at origin, return:
      - r_vec  : EPM position (in world frame)
      - mu_hat : world dipole direction (unit 3D vector, unconstrained)
      - R_epm  : rotation matrix with body y-axis = mu_hat
                 (no restriction to pure yaw; full 3D rotation).
    """

    # --- basic setup ---
    B_req = np.asarray(B_req, float)
    e_r   = np.asarray(e_r, float)
    e_r   = e_r / np.linalg.norm(e_r)

    Bmag = np.linalg.norm(B_req)
    if Bmag < 1e-12:
        raise ValueError("B must be non-zero")

    # --- dipole model matrix A: B ∝ A * mu_hat / r^3 ---
    A = 3.0 * np.outer(e_r, e_r) - np.eye(3)

    # --- unconstrained dipole direction from inverse of A ---
    B_hat = B_req / Bmag
    mu_hat_raw = np.linalg.solve(A, B_hat)   # some direction that maps to B_hat
    mu_hat = mu_hat_raw / np.linalg.norm(mu_hat_raw)  # unit vector, full 3D

    # --- solve for r to match |B| ---
    Ah = A @ mu_hat
    scale = np.linalg.norm(Ah)               # ||A * mu_hat||
    r_mag = ((mu0 * mu_mag * scale) / (4.0 * np.pi * Bmag)) ** (1.0 / 3.0)

    # place magnet so vector (magnet -> origin) = r_mag * e_r
    r_vec = -r_mag * e_r

    y_body = mu_hat

    # pick an "up" vector not parallel to y_body
    up = np.array([0.0, 0.0, -1.0])
    if abs(np.dot(up, y_body)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])

    # x_body = normalized (up × y_body)
    x_body = np.cross(up, y_body)
    x_body /= np.linalg.norm(x_body)

    # z_body = x_body × y_body
    z_body = np.cross(x_body, y_body)

    # # Enforce that body z-axis roughly points "up" in world frame
    # if z_body[2] < 0:
    #     x_body = -x_body
    #     z_body = -z_body

    R_epm = np.column_stack((x_body, y_body, z_body))


    # columns of R_epm are body axes expressed in world frame
    R_epm = np.column_stack((x_body, y_body, z_body))

    return r_vec, mu_hat, R_epm

def single_epm_pose_from_B(B_req, mu_mag, e_r, mu0=4e-7*np.pi):
    """
    Given required B at origin, return:
      - r_vec  : EPM position
      - mu_hat : world dipole direction (we enforce it in x–y plane)
      - R_epm  : rotation matrix with body y-axis = mu_hat,
                 and R_epm is a pure yaw about world z.
    """

    B_req = np.asarray(B_req, float)
    e_r   = np.asarray(e_r, float)
    e_r   = e_r / np.linalg.norm(e_r)

    Bmag = np.linalg.norm(B_req)
    if Bmag < 1e-12:
        raise ValueError("B must be non-zero")

    # Dipole model matrix
    A = 3.0 * np.outer(e_r, e_r) - np.eye(3)

    # First, unconstrained μ̃ from the dipole inverse
    B_hat = B_req / Bmag
    mu_hat_raw = np.linalg.solve(A, B_hat)
    mu_hat_raw /= np.linalg.norm(mu_hat_raw)

    # ---- NEW: force dipole into x–y plane and renormalise ----
    mu_eff = mu_hat_raw.copy()
    mu_eff[2] = 0.0  # kill z-component
    if np.linalg.norm(mu_eff[:2]) < 1e-8:
        # degenerate case: fall back to +y
        mu_eff = np.array([0.0, 1.0, 0.0])
    else:
        mu_eff /= np.linalg.norm(mu_eff)

    # Use this as our final world dipole direction
    mu_hat = mu_eff

    # Magnitude condition using this μ̂
    Ah = A @ mu_hat
    scale = np.linalg.norm(Ah)
    r_mag = ((mu0 * mu_mag * scale) / (4.0 * np.pi * Bmag)) ** (1.0 / 3.0)

    # Place magnet so that vector magnet→origin = r_mag * e_r
    r_vec = -r_mag * e_r

    # ---- NEW: build R_epm as *pure yaw about z* with y_body = μ_hat ----
    mx, my, _ = mu_hat
    psi = np.arctan2(-mx, my)   # yaw angle

    c, s = np.cos(psi), np.sin(psi)

    R_epm = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return r_vec, mu_hat, R_epm


# ======================
#  EPM pose for desired tip in y–z plane
# ======================

# def epm_pose_for_tip_yz(y, z,
#                         lengths, axes, masses,
#                         eta0_list, gvec, E, r, nu,
#                         L_cat, mu_mag, e_r, mu0=4e-7*np.pi,
#                         quad_n=12,
#                         prefer_B_dir=None):
#     # thetas_target = thetas_from_yz(y, z, lengths, L_cat)

#     B_req, info = magnetic_field_for_theta(
#         thetas_target, axes, lengths, masses, eta0_list,
#         f_e=np.zeros(6*len(lengths)),
#         gvec=gvec, E=E, r=r, nu=nu,
#         R_base=None, p_base=None, quad_n=quad_n,
#         prefer_dir=None,
#         reg=0.0
#     )

#     r_vec, mu_hat, R_epm = single_epm_pose_from_B(B_req, mu_mag, e_r, mu0)
#     return thetas_target, B_req, r_vec, mu_hat, R_epm, info
def solve_theta_given_B_lengths(thetas_init,
                                axes, lengths, masses, eta0_list,
                                B, f_e, gvec, E, r, nu,
                                R_base=None, p_base=None, quad_n=12):
    """
    Solve Eq. (20) for theta given:
      - B: magnetic field,
      - lengths, masses, etc.
    Uses your Gauss-Newton-LM solver.
    """
    theta_sol = my_gauss_newton_lm(
        thetas_init, axes, lengths, masses, eta0_list,
        B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
        R_base=R_base, p_base=p_base, quad_n=quad_n
    )
    return theta_sol
def epm_pose_for_tip_xy(x, y,
                        lengths, axes, masses,
                        eta0_list, gvec, E, r, nu,
                        L_cat, mu_mag, e_r, mu0=4e-7*np.pi,
                        quad_n=12,
                        prefer_B_dir=None):
    """
    Given a desired planar tip position (x,y) of the beam,
    solve (approximately) for the EPM pose that should produce
    that configuration under static equilibrium.

    Beam:
      - base at origin
      - initial axis along +y
      - bends in x–y plane about +z
    """

    # 1) x,y -> thetas  (constant curvature in x–y plane)
    # thetas_target = thetas_from_xy(x, y, lengths, L_cat)
    thetas_target = np.array([np.deg2rad(25)])
    # 2) thetas -> required B
    B_req, info = magnetic_field_for_theta(
        thetas_target, axes, lengths, masses, eta0_list,
        f_e=np.zeros(6*len(lengths)),
        gvec=gvec, E=E, r=r, nu=nu,
        R_base=None, p_base=None, quad_n=quad_n,
        prefer_dir=None,
        reg=0.0
    )

    # 3) B -> EPM pose from dipole model
    r_vec, mu_hat, R_epm = single_epm_pose_from_B_unrestricted(B_req, mu_mag, e_r, mu0)

    return thetas_target, B_req, r_vec, mu_hat, R_epm, info

# ======================
#  Beam & EPM parameters for your final block
# ======================

L_cat = 0.04
L1    = L_cat
r1    = 0.001

E_niti   = 3e6
rho_niti = 6700
nu_niti  = 0.33
gvec     = np.array([0.0, 0.0, -9.8])   # <<< CHANGED here too

A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0

lam    = rho_niti * A_cs
lengths = np.array([L1], float)
n       = len(lengths)
masses  = lam * lengths

r_arr   = np.array([r1],       float)
E_arr   = np.array([E_niti],   float)
nu_arr  = np.array([0.49],     float)
f_e     = np.zeros(6*n, float)

Br   = 1.4
mu0  = 4e-7 * np.pi
D_ipm = 15e-4
L_ipm = 4e-2
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm
mu_ipm_scalar = Br * V_ipm / mu0

# m_dir_local = np.array([0.0, 0.0, 1.0], float)
eta_ipm = (mu_ipm_scalar / L_ipm) * m_dir_local
eta0_list = [eta_ipm]

mu_mag = 250.0

# Magnet above the beam: beam along +y, origin at base. 
# "Above" = +z. e_r is direction FROM magnet TO origin.
# If magnet is at (0,0,+h) and origin at (0,0,0), vector from magnet to origin is (0,0,-1).
e_r    = np.array([0.0, 0.0, -1.0])   # <<< CHANGED: magnet above (+z)

# desired tip position in (y,z) plane
# desired tip position in (x,y) plane
x_target = 20e-3    # sideways deflection
y_target = 35e-3    # along the beam

thetas_target, B_req, r_epm, mu_hat_epm, R_epm, info = epm_pose_for_tip_xy(
    x_target, y_target,
    lengths=lengths, axes=axes, masses=masses,
    eta0_list=eta0_list, gvec=gvec,
    E=E_arr, r=r_arr, nu=nu_arr,
    L_cat=L_cat,
    mu_mag=mu_mag, e_r=e_r,
    mu0=mu0, prefer_B_dir = prefer_dir
)


print("thetas_target (deg):", np.rad2deg(thetas_target))
print("Required B (T):", B_req, " |B| (mT):", np.linalg.norm(B_req)*1e3)
print("EPM position r_epm:", r_epm, " |r|:", np.linalg.norm(r_epm))
print("EPM dipole direction μ̂:", mu_hat_epm)
def tip_pos_from_B_s(x,               # x = [Bx, By, Bz, s]
                     thetas_init,
                     axes, lengths0, lam,          # lam = rho_niti * A_cs
                     eta0_list, gvec, E_arr, r_arr, nu_arr,
                     f_e=None,
                     R_base=None, p_base=None,
                     quad_n=12):
    """
    Map control variables x = [B_x, B_y, B_z, s] to tip position p_tip.

    lengths0 : nominal lengths array [L1, L2, ...]
    lam      : linear mass density (rho * area)
    """

    x = np.asarray(x, float)
    B = x[:3]
    s = float(x[3])

    # 1) Modify lengths: L1 -> L1 + s
    lengths = np.asarray(lengths0, float).copy()
    lengths[0] = lengths0[0] + s

    # 2) Update masses for each segment
    masses = lam * lengths

    # 3) Solve equilibrium for theta with these lengths and B
    if f_e is None:
        f_e = np.zeros(6*len(lengths), float)

    theta_sol = solve_theta_given_B_lengths(
        thetas_init=thetas_init,
        axes=axes, lengths=lengths, masses=masses,
        eta0_list=eta0_list,
        B=B, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
        R_base=R_base, p_base=p_base, quad_n=quad_n
    )

    # 4) Compute tip position from theta_sol
    p_tip = tip_position_from_theta(
        axes, theta_sol, lengths,
        R_base=R_base, p_base=p_base
    )
    return p_tip


print(repr(R_epm))
print(thetas_target)
# thetas_target2 = np.array([np.deg2rad(1)])
thetas_eq = my_gauss_newton_lm(
    thetas_target, axes, lengths, masses, eta0_list,
    B=B_req, f_e=f_e, gvec=gvec, E=E_arr, r=r_arr, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12
)
s0 = 0.0 
x0 = np.hstack([B_req, s0]) 
print(f"Bending is {np.rad2deg(thetas_eq)}")
p_tip = tip_pos_from_B_s( 
    x0, thetas_init=thetas_eq, axes=axes, lengths0=lengths, lam=lam, 
    eta0_list=eta0_list, gvec=gvec, E_arr=E_arr, r_arr=r_arr, 
    nu_arr=nu_arr, f_e=f_e, R_base=None, p_base=None, quad_n=12 ) 
print("Tip position p_tip:", p_tip)


mu_world_from_R = R_epm @ np.array([0.0, 1.0, 0.0])
# mu_world_from_R == mu_hat  (up to numerical noise)
mu_check = R_epm @ np.array([0.0, 1.0, 0.0])
print("μ_hat from solver:", mu_world_from_R)
print("μ from R_epm @ e_y:", mu_check)



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
# import numpy as np

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
# import numpy as np

# def plot_beam_and_epm(axes_list, thetas, lengths,
#                       r_epm, mu_hat_epm, R_epm,
#                       B_req=None,
#                       R_base=None, p_base=None,
#                       cyl_radius=0.15,
#                       cyl_length=0.9):
#     """
#     Visualise:
#       - beam centerline,
#       - external cylindrical magnet:
#           * axis along WORLD Z,
#           * r_epm = magnet CENTRE (same as dipole position),
#           * half red / half blue on circular cross-section,
#             split rotated to match the dipole direction
#             (projection of μ̂ onto x–y plane),
#       - EPM dipole direction (μ̂),
#       - (optionally) B-field at origin,
#       - (optionally) EPM local axes from R_epm.
#     """

#     # Get beam nodes from stack_blocks
#     Rs_si, ps_si, J, R_nodes, p_nodes = stack_blocks(
#         axes_list, thetas, lengths,
#         s_list=lengths,
#         R_base=R_base, p_base=p_base
#     )
#     p_nodes = np.asarray(p_nodes)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # --- Plot beam centerline ---
#     ax.plot(p_nodes[:, 0], p_nodes[:, 1], p_nodes[:, 2],
#             '-o', label='Beam', linewidth=2)

#     # Mark base and tip
#     ax.scatter(p_nodes[0, 0], p_nodes[0, 1], p_nodes[0, 2],
#                color='k', s=40, label='Base')
#     ax.scatter(p_nodes[-1, 0], p_nodes[-1, 1], p_nodes[-1, 2],
#                color='r', s=40, label='Tip')

#     # --- EPM position ---
#     r_epm = np.asarray(r_epm, float)
#     mu_hat_epm = np.asarray(mu_hat_epm, float)
#     R_epm = np.asarray(R_epm, float)

#     ax.scatter(r_epm[0], r_epm[1], r_epm[2],
#                color='magenta', s=60, label='Magnet center / dipole')

#     # ---------- External magnet as cylinder aligned with WORLD Z ----------
#     # r_epm is the *center* of the cylinder.
#     # Cylinder axis: world z
#     n_theta = 80
#     n_h = 10

#     theta = np.linspace(0.0, 2*np.pi, n_theta)
#     h = np.linspace(-cyl_length/2.0, cyl_length/2.0, n_h)  # centered at r_epm.z
#     theta_grid, h_grid = np.meshgrid(theta, h)

#     # Cylinder geometry in world frame
#     Xc = r_epm[0] + cyl_radius * np.cos(theta_grid)
#     Yc = r_epm[1] + cyl_radius * np.sin(theta_grid)
#     Zc = r_epm[2] + h_grid

#     # --- Color pattern: half red / half blue aligned with μ direction ---
#     # Project μ onto x–y plane:
#     mu_xy = np.array([mu_hat_epm[0], mu_hat_epm[1]])
#     mu_xy_norm = np.linalg.norm(mu_xy)

#     if mu_xy_norm < 1e-8:
#         # If μ is almost vertical, choose some default split (e.g. along +x/-x)
#         mu_xy_hat = np.array([1.0, 0.0])
#     else:
#         mu_xy_hat = mu_xy / mu_xy_norm

#     # Radial direction for each (theta) on the cylinder surface
#     r_x = np.cos(theta_grid)
#     r_y = np.sin(theta_grid)

#     # Dot product between projected μ and radial direction
#     # > 0 -> "north" side (red), < 0 -> "south" side (blue)
#     dots = mu_xy_hat[0] * r_x + mu_xy_hat[1] * r_y

#     colors = np.empty(theta_grid.shape + (4,), dtype=float)
#     red_rgba  = np.array([1.0, 0.0, 0.0, 0.5])
#     blue_rgba = np.array([0.0, 0.0, 1.0, 0.5])

#     colors[dots >= 0] = red_rgba
#     colors[dots <  0] = blue_rgba

#     ax.plot_surface(Xc, Yc, Zc,
#                     facecolors=colors,
#                     linewidth=0.2,
#                     edgecolor='k')

#     # --- EPM dipole direction as arrow ---
#     L_mu = cyl_length * 0.7
#     mu_end = r_epm + L_mu * mu_hat_epm
#     ax.plot([r_epm[0], mu_end[0]],
#             [r_epm[1], mu_end[1]],
#             [r_epm[2], mu_end[2]],
#             '-', linewidth=3, label='μ̂ (EPM dipole)')

#     # --- EPM local axes from R_epm (this is where that matrix you printed shows up) ---
#     L_axis = cyl_length * 0.5
#     origin = r_epm
#     x_body = origin + L_axis * R_epm[:, 0]
#     y_body = origin + L_axis * R_epm[:, 1]
#     z_body = origin + L_axis * R_epm[:, 2]

#     ax.plot([origin[0], x_body[0]],
#             [origin[1], x_body[1]],
#             [origin[2], x_body[2]],
#             '-', label='EPM x_body')
#     ax.plot([origin[0], y_body[0]],
#             [origin[1], y_body[1]],
#             [origin[2], y_body[2]],
#             '-', label='EPM y_body')
#     ax.plot([origin[0], z_body[0]],
#             [origin[1], z_body[1]],
#             [origin[2], z_body[2]],
#             '-', label='EPM z_body')

#     # --- B field at origin (optional) ---
#     if B_req is not None:
#         B_req = np.asarray(B_req, float)
#         L_B = cyl_length * 0.7
#         B_hat = B_req / (np.linalg.norm(B_req) + 1e-12)
#         B_end = L_B * B_hat

#         ax.plot([0.0, B_end[0]],
#                 [0.0, B_end[1]],
#                 [0.0, B_end[2]],
#                 '-', linewidth=2, label='B direction @ origin')

#     # Axes labels
#     ax.set_xlabel('x [m]')
#     ax.set_ylabel('y [m]')
#     ax.set_zlabel('z [m]')

#     # Equal aspect ratio (include cylinder in bounds)
#     xs = np.concatenate([p_nodes[:, 0], Xc.ravel(), [r_epm[0]]])
#     ys = np.concatenate([p_nodes[:, 1], Yc.ravel(), [r_epm[1]]])
#     zs = np.concatenate([p_nodes[:, 2], Zc.ravel(), [r_epm[2]]])

#     x_range = xs.max() - xs.min()
#     y_range = ys.max() - ys.min()
#     z_range = zs.max() - zs.min()
#     max_range = max(x_range, y_range, z_range) * 0.6

#     x_mid = 0.5 * (xs.max() + xs.min())
#     y_mid = 0.5 * (ys.max() + ys.min())
#     z_mid = 0.5 * (zs.max() + zs.min())

#     ax.set_xlim(x_mid - max_range, x_mid + max_range)
#     ax.set_ylim(y_mid - max_range, y_mid + max_range)
#     ax.set_zlim(z_mid - max_range, z_mid + max_range)

#     ax.legend()
#     ax.set_title('Beam + External Cylindrical Magnet (μ-aligned split, axis = z)')

#     plt.tight_layout()
#     plt.show()

# plot_beam_and_epm(
#     axes_list=axes,
#     thetas=thetas_eq,
#     lengths=lengths,
#     r_epm=r_epm,    
#     mu_hat_epm=mu_hat_epm,
#     R_epm=R_epm,
#     B_req=B_req,
#     R_base=None,
#     p_base=None,
#     cyl_radius=0.15,
#     cyl_length=0.2,
# )

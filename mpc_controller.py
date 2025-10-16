# mpc_controller.py
import numpy as np
import scipy.sparse as sp
import osqp
from scipy.optimize import linprog
from scipy.linalg import solve_discrete_are, block_diag

# ---- small utilities (lifted from your sim, minimally tweaked) ----

def dare_stabilizing_K(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P

def trust_radius(jac_fn, psi_rad, *,
                 h_rad=np.deg2rad(0.5),
                 eps_theta_rad=np.deg2rad(1.0),
                 Jmin=1e-6, Lmin=1e-6,
                 dpsi_cap=np.deg2rad(5.0)):
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
    n, m = B_list[0].shape
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))

    P_prev = np.eye(n)
    for i in range(N):
        P_prev = Phi_list[i] @ P_prev
        Mx[i*n:(i+1)*n, :] = P_prev

    for i in range(N):
        for j in range(i+1):
            if i == j:
                Tij = np.eye(n)
            else:
                Tij = np.eye(n)
                for s in range(i, j, -1):
                    Tij = Phi_list[s] @ Tij
            Mc[i*n:(i+1)*n, j*m:(j+1)*m] = Tij @ B_list[j]
    return Mx, Mc

def solve_qp_osqp(H, f, A, l, u, U_warm=None):
    P = sp.csc_matrix(0.5*(H + H.T))
    q = f.astype(float)
    A = sp.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    if U_warm is not None:
        prob.warm_start(x=U_warm)
    res = prob.solve()
    status = res.info.status
    if status not in ("solved", "solved inaccurate"):
        return None, None, status
    return res.x, res.y, status

def compute_VT_MPI(A, B, F, G, K=None, Q=None, R=None, tol=1e-8, nu_max=200):
    n = A.shape[0]
    if K is None:
        if (Q is None) or (R is None):
            raise ValueError("Provide K, or Q and R to compute it.")
        K, _ = dare_stabilizing_K(A, B, Q, R)
    Phi = A + B @ K
    M = np.vstack([F, G @ K])
    V_stack = M.copy()
    Phi_pow = np.eye(n)

    for nu in range(0, nu_max+1):
        Phi_pow = Phi_pow @ Phi
        W = M @ Phi_pow
        ok_all = True
        for j in range(W.shape[0]):
            c = -W[j]
            A_ub = V_stack
            b_ub = np.ones(V_stack.shape[0])
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs")
            if (res.status != 0):
                raise RuntimeError("LP failed while checking MPI condition.")
            max_val = -res.fun
            if max_val > 1.0 + tol:
                ok_all = False
                break
        if ok_all:
            V_T = []
            Phi_pow_i = np.eye(n)
            for i in range(nu+1):
                V_T.append(M @ Phi_pow_i)
                Phi_pow_i = Phi_pow_i @ Phi
            return np.vstack(V_T), nu
        V_stack = np.vstack([V_stack, W])

    raise RuntimeError("Reached nu_max without satisfying the MPI condition.")

# ---- main controller ------------------------------------------------

class MPCController:
    """
    SISO TV-MPC wrapper for real-time use.
    - State x = [θ] (rad)
    - Control u = ψ̇ (rad/s)
    - Plant map θ = theta_fn(ψ), J = dθ/dψ = J_fn(ψ)
    - Robot command is absolute ψ (your joint 6), updated as ψ_{k+1} = ψ_k + u_0*dt
    """
    def __init__(
        self,
        *,
        theta_fn,               # callable: θ = theta_fn(ψ)
        J_fn,                   # callable: J = dθ/dψ = J_fn(ψ)
        dt=0.1,
        Np=10,
        w_th=10.0,
        w_u=5e-2,
        theta_band_deg=10.0,    # soft band around θ_ref for constraints (set np.inf to disable)
        eps_theta_deg=10.0,     # trust-region sizing parameter
        h_deg_for_radius=0.5,   # finite-diff step for J'
        trust_region_deg=180.0, # hard cap per-step |Δψ| bound used inside trust calc
        theta_max_deg=90.0,     # state polytope bound
        u_max_deg_s=90.0,       # input polytope bound
        j6_min_rad=-2.9,        # actuator absolute limits
        j6_max_rad=+2.9,
        rate_limit_deg=15.0     # optional actuator rate limit |Δψ|/step
    ):
        self.theta_fn = theta_fn
        self.J_fn = J_fn
        self.dt = float(dt)
        self.Np = int(Np)
        self.Q = np.array([[w_th]], float)
        self.R = np.array([[w_u]], float)
        self.band_deg = float(theta_band_deg)
        self.eps_theta_deg = float(eps_theta_deg)
        self.h_deg_for_radius = float(h_deg_for_radius)
        self.trust_region = np.deg2rad(trust_region_deg)
        self.theta_max = np.deg2rad(theta_max_deg)
        self.u_max = np.deg2rad(u_max_deg_s)
        self.j6_min = float(j6_min_rad)
        self.j6_max = float(j6_max_rad)
        self.rate_limit = np.deg2rad(rate_limit_deg)

        # 1-state canonical A, B are recomputed per step (B uses J(ψ)*dt)
        self.A = np.array([[1.0]])
        self.psi = 0.0               # internal estimate of current ψ (rad)
        self.U_prev = np.zeros(self.Np)  # warm start
        self.Qf = None               # terminal weight (set per step from DARE with B_N-1)
        self.V_T = None              # terminal set (recomputed as needed)

        # Precompute triangular integration matrix for ψ (S @ U = Δψ horizon)
        self.S_np = np.tril(np.ones((self.Np, self.Np), float)) * self.dt

    def set_initial_psi(self, psi0_rad: float):
        self.psi = float(psi0_rad)

    def _tv_gain_sequence(self, B_list):
        """Backward Riccati with time-varying B_k -> K_seq, Qf if needed."""
        P_next = self.Qf if self.Qf is not None else solve_discrete_are(self.A, B_list[-1], self.Q, self.R)
        K_seq = [None]*self.Np
        for k in range(self.Np-1, -1, -1):
            Bk = B_list[k]
            S  = self.R + Bk.T @ P_next @ Bk
            Kk = -np.linalg.solve(S, Bk.T @ P_next @ self.A)
            K_seq[k] = Kk
            Acl = self.A + Bk @ Kk
            P_next = self.Q + self.A.T @ P_next @ Acl
        if self.Qf is None:
            self.Qf = P_next  # last computed P acts like terminal weight
        return K_seq

    def _build_horizon_linearisation(self, psi_now):
        # shift + hold-last nominal plan
        U_nom = np.r_[self.U_prev[1:], self.U_prev[-1]] if self.U_prev.size else np.zeros(self.Np)
        psi_nom = psi_now + self.S_np @ U_nom

        # TV B_list (1x1 each) and per-step trust dpsi bound
        B_list = []
        dpsi_vec = np.zeros(self.Np)
        for i in range(self.Np):
            Ji = float(self.J_fn(psi_nom[i]))
            # sign-preserving floor to avoid singular B
            Ji = np.sign(Ji) * max(abs(Ji), 1e-6)
            B_list.append(np.array([[self.dt * Ji]]))
            dpsi_vec[i] = trust_radius(
                self.J_fn, psi_nom[i],
                h_rad=np.deg2rad(self.h_deg_for_radius),
                eps_theta_rad=np.deg2rad(self.eps_theta_deg),
                Jmin=1e-6, Lmin=1e-6, dpsi_cap=self.trust_region
            )
        return psi_nom, U_nom, B_list, dpsi_vec

    def _ensure_terminal_set(self, B_last):
        # state/input polytope: |θ|<=θ_max, |u|<=u_max
        F_state = np.array([[ 1.0/self.theta_max],
                            [-1.0/self.theta_max]])
        G_input = np.array([[ 1.0/self.u_max],
                            [-1.0/self.u_max]])
        K_dare, P_dare = dare_stabilizing_K(self.A, B_last, self.Q, self.R)
        self.Qf = P_dare
        self.V_T, _ = compute_VT_MPI(self.A, B_last, F_state, G_input, K=K_dare)

    def step(self, ref_theta_rad: float, theta_meas_rad: float):
        """
        One RT cycle.
        Inputs:
          - ref_theta_rad: desired θ at current step (rad)
          - theta_meas_rad: measured θ (rad) from vision
        Returns:
          - j6_cmd_rad: absolute ψ command for joint 6 (clamped & rate-limited)
          - info: diagnostics dict
        """
        # time-varying linearisation around current ψ and warm-start U_prev
        psi_k = float(self.psi)
        psi_nom, U_nom, B_list, dpsi_vec = self._build_horizon_linearisation(psi_k)
        self._ensure_terminal_set(B_list[-1])

        # gain sequence and closed-loop Φ_k
        K_seq = self._tv_gain_sequence(B_list)
        Phi_list = [self.A + B_list[k] @ K_seq[k] for k in range(self.Np)]

        # stacked prediction matrices
        Mx, Mc = seq_mat_tv(Phi_list, B_list)
        Kbar = block_diag(*K_seq)

        # costs (same structure as your sim)
        n, m = 1, 1
        Qtil = np.zeros((self.Np*n, self.Np*n))
        if self.Np > 1:
            Qtil[:(self.Np-1)*n, :(self.Np-1)*n] = np.kron(np.eye(self.Np-1), self.Q)
        Qtil[(self.Np-1)*n:, (self.Np-1)*n:] = self.Qf
        Rtil = np.kron(np.eye(self.Np), self.R)

        xk = np.array([theta_meas_rad], float)   # use measurement for state
        X0 = Mx @ xk
        KC = Kbar @ Mc
        IUm = np.eye(self.Np*m)
        U0 = Kbar @ X0

        # constant ref horizon (hold-last) — adapt if you have a ref stream
        xref_seq = np.full((self.Np,), float(ref_theta_rad))
        H = 2.0 * (Mc.T @ Qtil @ Mc + (KC + IUm).T @ Rtil @ (KC + IUm))
        f = 2.0 * (Mc.T @ Qtil @ (X0 - xref_seq) + (KC + IUm).T @ Rtil @ U0)
        H = 0.5 * (H + H.T)

        # constraints
        A_osqp = np.empty((0, self.Np*m)); l_osqp = np.empty(0); u_osqp = np.empty(0)

        # (A) ψ tube: |ψ_pred - ψ_nom| ≤ dpsi_bound
        J_u   = Kbar @ Mc + IUm
        A_tube = self.S_np @ J_u
        offset = self.S_np @ (U0 - U_nom)
        dpsi_bound = np.maximum(dpsi_vec, 1e-12)
        l_tube = -dpsi_bound - offset
        u_tube = +dpsi_bound - offset
        A_osqp = np.vstack([A_osqp, A_tube])
        l_osqp = np.concatenate([l_osqp, l_tube])
        u_osqp = np.concatenate([u_osqp, u_tube])

        # (B) θ band (optional)
        if np.isfinite(self.band_deg):
            band = np.deg2rad(self.band_deg)
            e_theta = np.array([[1.0]])
            E   = np.kron(np.eye(self.Np), e_theta)
            EX0 = E @ X0
            theta_ref = xref_seq
            A_th  = E @ Mc
            rhs_p = theta_ref + band - EX0
            rhs_n = -theta_ref + band + EX0

            A_osqp = np.vstack([A_osqp,  A_th,  -A_th])
            l_osqp = np.concatenate([l_osqp, -np.inf*np.ones(self.Np), -np.inf*np.ones(self.Np)])
            u_osqp = np.concatenate([u_osqp,  rhs_p,                    rhs_n])

        # (C) terminal set
        Sn = np.zeros((1, self.Np*1)); Sn[:, (self.Np-1)*1:self.Np*1] = np.eye(1)
        A_term = self.V_T @ (Sn @ Mc)
        u_term = 1.0 - (self.V_T @ (Sn @ X0)).ravel()
        A_osqp = np.vstack([A_osqp, A_term])
        l_osqp = np.concatenate([l_osqp, -np.inf*np.ones_like(u_term)])
        u_osqp = np.concatenate([u_osqp,  u_term])

        # solve QP
        c_opt, y, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=self.U_prev)
        infeas = (status not in ("solved", "solved inaccurate")) or (c_opt is None)

        if infeas:
            # fallback: hold ψ (u0 = 0)
            u0 = 0.0
            U_tr = np.zeros(self.Np)
        else:
            X_pred = X0 + Mc @ c_opt
            U_pred = Kbar @ X_pred + c_opt
            U_tr = np.asarray(U_pred).reshape(-1)
            u0 = float(U_tr[0])

        # integrate to next ψ command
        psi_cmd = self.psi + u0 * self.dt

        # actuator safety: clamp and rate limit
        psi_cmd = float(np.clip(psi_cmd, self.j6_min, self.j6_max))
        psi_cmd = float(np.clip(psi_cmd, self.psi - self.rate_limit, self.psi + self.rate_limit))

        # update internals for next cycle
        self.U_prev = U_tr if not infeas else np.zeros_like(self.U_prev)
        self.psi = psi_cmd

        info = dict(
            status=status,
            u0_rad_s=u0,
            psi_cmd_rad=psi_cmd,
            psi_now_rad=psi_k,
            infeasible=int(infeas),
        )
        return psi_cmd, info

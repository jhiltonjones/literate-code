import numpy as np 
from mpc_controller_functions import dare_stabilising_K, seq_mat_lti, solve_qp_osqp
class mpc_controller_LTI_nocorr:
    def __init__(self, *, J_fn, dt=0.0, Np=10,
                 w_th=10.0, w_u=5e-2,
                 theta_band_deg=10.0,
                 theta_max_deg=90,
                 u_max_deg_s=90):

        self.J_fn = J_fn
        self.dt = float(dt)
        self.Np = int(Np)
        self.Q = np.array([[w_th]])
        self.R = np.array([[w_u]])
        self.band_deg = float(theta_band_deg)
        self.theta_max = np.deg2rad(theta_max_deg)
        self.u_max = np.deg2rad(u_max_deg_s)
        self.A = np.array([[1.0]])
        self.Qf = None

    def set_intial_psi(self, psi0_rad):
        self.psi = float(psi0_rad)

    def set_dt(self, new_dt):
        self.dt = float(new_dt)
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt

    def _build_lti_model(self, psi_now):

        J0 = float(self.J_fn(psi_now))
        J0 = np.sign(J0)*max(abs(J0), 1e-6)
        B = np.array([[self.dt*J0]])
        
        _, P = dare_stabilising_K(self.A, B, self.Q, self.R)
        self.Qf = P
        Mx, Mc = seq_mat_lti(self.A, B, self.Np)
        return B, Mx, Mc
    
    def step(self, ref_seq_rad, theta_meas_rad):
        psi_k = float(self.psi)
        B, Mx, Mc = self._build_lti_model(psi_k)
        n,m = 1,1   
        Np = self.Np
        Qtil = np.zeros((Np*n, Np*n))
        if Np > 1:
            Qtil[:(Np-1)*n,:(Np-1)*n] = np.kron(np.eye(Np-1), self.Q)
        Qtil[(Np-1)*n:, (Np-1)*n:] =self.Qf
        Rtil = np.kron(np.eye(Np), self.R)
        xk = np.array([[theta_meas_rad]])
        xref_seq = np.asarray(ref_seq_rad).reshape(Np,1)
        xref_stack = xref_seq.reshape(Np * n, 1)
        X0_stack = (Mx @ xk).reshape(Np * n, 1)
        H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil)
        f = 2.0 * (Mc.T @ Qtil @ (X0_stack - xref_stack))

        A_list = []
        l_list = []
        u_list = []

        if np.isfinite(self.u_max):
            A_u = np.eye(Np * m)
            u_max_vec = self.u_max * np.ones(Np * m)
            A_list.append(A_u)
            l_list.append(-u_max_vec)
            u_list.append(+u_max_vec)
            if np.isfinite(self.band_deg):
                band = np.deg2rad(self.band_deg)
                band_vec = band * np.ones(Np * n)

                rhs_p = (band_vec + xref_stack - X0_stack).ravel()

                rhs_n = (band_vec - xref_stack + X0_stack).ravel()

                A_list.append( Mc)
                l_list.append(-np.inf * np.ones(Np * n))
                u_list.append(rhs_p)

                A_list.append(-Mc)
                l_list.append(-np.inf * np.ones(Np * n))
                u_list.append(rhs_n)

        if A_list:
            A_osqp = np.vstack(A_list)
            l_osqp = np.concatenate(l_list)
            u_osqp = np.concatenate(u_list)
        else:
            A_osqp = np.zeros((0, Np * m))
            l_osqp = np.zeros(0)
            u_osqp = np.zeros(0)

        u_opt, y, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=None)
        infeas = (status not in ("solved", "solved inaccurate")) or (u_opt is None)

        if infeas:
            u_seq = np.zeros(Np)
            u0 = 0.0
            X_pred = np.full((Np, 1), np.nan)
        else:
            u_seq = np.asarray(u_opt).ravel()
            X_pred = (X0_stack + Mc @ u_opt.reshape(-1, 1)).reshape(Np, 1)
            u0 = float(u_seq[0])

        psi_cmd = self.psi + u0 * self.dt
        self.psi = psi_cmd

        info = dict(
            status=status,
            infeasible=int(infeas),
            u0_rad_s=u0,
            psi_now_rad=float(psi_k),
            psi_cmd_rad=psi_cmd,
            xref_seq_rad=xref_seq.ravel().copy(),
            theta_pred_rad=X_pred.ravel().copy(),
            u_seq_rad_s=u_seq.copy(),
        )
        return psi_cmd, info

    
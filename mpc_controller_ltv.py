import numpy as np
from scipy.linalg import solve_discrete_are, block_diag

from mpc_controller_functions import seq_mat_tv, solve_qp_osqp, trust_radius

class MPC_controller_time_varying:
    def __init__(self, *, J_fn, dt = 0.0, Np = 10, w_th = 10.0, w_u = 5e-2, theta_band_deg=10.0, eps_theta_deg = 10.0, h_deg_for_radius = 0.5,
                 trust_region_deg = 180, theta_max_deg = 90, u_max_deg_s = 90, rate_limit_deg = 15.0):
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
        self.rate_limit = np.deg2rad(rate_limit_deg)
        self.A = np.array([[1.0]])
        self.Qf = None
        self.V_T = None
        self.S_np = np.tril(np.ones((self.Np, self.Np)))*self.dt #This is a ZOH which assumes that the input u will be held constant over its interval

    def set_intial_psi(self, psi0_rad):
        self.psi = float(psi0_rad)
        
    def set_dt(self, new_dt):
        self.dt = float(new_dt)
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt
    def _tv_gain_sequence(self, B_list):
        P_next = self.Qf if self.Qf is not None else solve_discrete_are(self.A, B_list[-1], self.Q, self.R)
        K_seq = [None]* self.Np
        for k in range(self.Np-1, -1, -1):
            Bk = B_list[k]
            S = self.R +Bk.T @ P_next @ Bk
            Kk = -np.linalg.solve(S, Bk.T @ P_next @ self.A)
            K_seq[k] = Kk
            Acl = self.A + Bk @ Kk
            P_next = self.Q + Acl.T @ P_next @ Acl + Kk.T @ self.R @ Kk
        if self.Qf is None:
            self.Qf = P_next
        return K_seq
    
    def _build_horizon_linearisation(self, psi_now):
        if not hasattr(self, "U_prev"):
            self.U_prev = np.zeros(self.Np)
        U_nom = np.roll(self.U_prev, -1) #This shifts the previous control inputs left to reuse the starting guess for our linearisation
        U_nom[-1] = U_nom[-2] if self.Np >1 else U_nom[-1]
        psi_nom = psi_now + self.S_np @ U_nom
        B_list = []
        dpsi_vec = np.zeros(self.Np)

        for i in range(self.Np):
            Ji = float(self.J_fn(psi_nom[i]))
            Ji = np.sign(Ji) * max(abs(Ji), 1e-6)
            B_list.append(np.array([[self.dt*Ji]]))
            dpsi_vec[i] = trust_radius(
                self.J_fn, psi_nom[i],
                h_rad=np.deg2rad(self.h_deg_for_radius),
                eps_theta_rad=np.deg2rad(self.eps_theta_deg),
                Jmin=1e-6, Lmin=1e-6, dpsi_cap=self.trust_region
            )
        return psi_nom, U_nom, B_list, dpsi_vec
    
    def step_with_seq(self, ref_seq_rad, theta_meas_rad):
        use_tube = False

        psi_k = float(self.psi)
        psi_nom, U_nom, B_list, dpsi_vec = self._build_horizon_linearisation(psi_k)
        K_seq = self._tv_gain_sequence(B_list)
        Phi_list = [self.A + B_list[k] @ K_seq[k] for k in range(self.Np)]
        Mx, Mc = seq_mat_tv(Phi_list, B_list)
        Kbar = block_diag(*K_seq)

        n, m = 1, 1
        Qtil = np.zeros((self.Np*n, self.Np*n))
        if self.Np > 1:
            Qtil[:(self.Np-1)*n, :(self.Np-1)*n] = np.kron(np.eye(self.Np-1), self.Q)
        Qtil[(self.Np-1)*n:, (self.Np-1)*n:] = self.Qf
        Rtil = np.kron(np.eye(self.Np), self.R)
        xk = np.array([[theta_meas_rad]]) 
        X0 = (Mx @ xk).reshape(self.Np, 1)  #This gives the current state prediction without corrections
        KC = Kbar @ Mc  #Tells us how the corrections will affect the output                                    
        IUm = np.eye(self.Np*m)
        U0 = (Kbar @ X0)  #This is the input without corrections
              

        xref_seq = np.asarray(ref_seq_rad, float).reshape(self.Np, 1) 
        J_u = KC + IUm
        H = 2.0 * (Mc.T @ Qtil @ Mc + (J_u).T @ Rtil @ (J_u))
        f = 2.0 * (Mc.T @ Qtil @ (X0 - xref_seq) + (KC + IUm).T @ Rtil @ U0) 
        A_osqp = np.empty((0, self.Np*m)); l_osqp = np.empty(0); u_osqp = np.empty(0)
        A_tube = self.S_np @ J_u                       
        offset = (self.S_np @ (U0 - U_nom.reshape(self.Np,1))).reshape(self.Np) 
        if use_tube:
            # (tube / trust region on cumulative Δψ)
            dpsi_bound = max(float(dpsi_vec), 1e-12)            # or whatever you used
            # J_u must be defined earlier as Kbar @ Mc + IUm, etc.
            A_tube  = self.S_np @ J_u                       # (Np, Np)
            if not hasattr(self, "U_prev"):
                self.U_prev = np.zeros(self.Np)
            U_nom   = np.roll(self.U_prev, -1)
            if self.Np > 1:
                U_nom[-1] = U_nom[-2]

            offset  = (self.S_np @ (U0 - U_nom.reshape(self.Np, 1))).reshape(self.Np)
            l_tube  = -dpsi_bound - offset
            u_tube  = +dpsi_bound - offset

            A_osqp  = np.vstack([A_osqp, A_tube])
            l_osqp  = np.concatenate([l_osqp, l_tube])
            u_osqp  = np.concatenate([u_osqp, u_tube])

        if np.isfinite(self.band_deg):
            band = np.deg2rad(self.band_deg)
            e_theta = np.array([[1.0]])
            E   = np.kron(np.eye(self.Np), e_theta)    
            EX0 = (E @ X0).reshape(self.Np, 1)                 
            A_th  = E @ Mc                                      
            rhs_p = (xref_seq + band - EX0).reshape(self.Np)   
            rhs_n = (-xref_seq + band + EX0).reshape(self.Np)  

            A_osqp = np.vstack([A_osqp,  A_th,  -A_th])
            l_osqp = np.concatenate([l_osqp, -np.inf*np.ones(self.Np), -np.inf*np.ones(self.Np)])
            u_osqp = np.concatenate([u_osqp,  rhs_p, rhs_n])
        if getattr(self, "V_T", None) is not None and self.V_T.size > 0:
            Sn = np.zeros((1, self.Np*1))
            Sn[:, (self.Np-1)*1:self.Np*1] = np.eye(1)

            A_term = self.V_T @ (Sn @ Mc)                 
            u_term = 1.0 - (self.V_T @ (Sn @ X0)).ravel()     

            A_osqp = np.vstack([A_osqp, A_term])
            l_osqp = np.concatenate([l_osqp, -np.inf*np.ones_like(u_term)])
            u_osqp = np.concatenate([u_osqp,  u_term])    
        c_opt, y, status = solve_qp_osqp(H,f, A_osqp, l_osqp, u_osqp, U_warm=None)   

        infeas = (status not in ("solved" , "solved inaccurate")) or (c_opt is None)

        if infeas:
            print("INFEASIBLE")
            u0 = 0.0    
            U_tr = np.zeros(self.Np)
            X_pred = np.full((self.Np,1), np.nan)
            psi_pred = np.full((self.Np,), np.nan)   
        else:
            c_col = np.asarray(c_opt).reshape(self.Np,1)
            X_pred = (X0 +Mc @c_col).reshape(self.Np,1)
            U_pred = (Kbar @ X_pred + c_col).ravel()
            U_tr = U_pred
            u0 = float(U_tr[0])
            psi_pred = psi_k +( self.S_np @ U_tr)


        psi_cmd = self.psi + u0*self.dt

        self.U_prev = U_tr if not infeas else np.zeros_like(self.U_prev)
        self.psi = psi_cmd        
        info = dict(
            status=status,
            infeasible=int(infeas),
            u0_rad_s=u0,
            psi_now_rad=float(psi_k),
            psi_cmd_rad=psi_cmd,
            xref_seq_rad=xref_seq.ravel().copy(),        
            theta_pred_rad=X_pred.ravel().copy(),        
            u_seq_rad_s=(U_tr.copy() if not infeas else np.full(self.Np, np.nan)),
            psi_nom_rad=psi_nom.copy(),                
            psi_pred_rad=psi_pred.copy(),              
            # dpsi_bound_rad=dpsi_bound.copy(),         
        )
        return psi_cmd, info
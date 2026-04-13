import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Cantiliver_Beam_Solver:
    # We initialize the parameters of the beam and we calculete matrices
    def __init__(self, E, I, L, n, A, rho):
        self.E = E
        self.I = I
        self.L = L
        self.n = n
        self.A = A
        #The density of the material
        self.rho = rho

        # Element length
        self.le = L / n

        # the number of nodes
        self.n_node = n + 1
        # the number of degrees of freedom each node has (2 for 1D beam element)
        self.node_DOF = 2
        # The total number of degrees of freedom
        self.DOF = self.node_DOF * self.n_node

        # maximum axial load
        self.axial_force = 0.0      
        # The perturbation force
        self.lateral_force = 0.5

        # Rayleigh damping coefficients ( C= alpha * M + beta * K )
        self.alpha = 0.04
        self.beta_rayleigh = 0.04
        self.max_time=0
        #Stability indicator
        self.stable=True
        self.st=0
        #The cretical load (when the structure starts loosing stability)
        self.cretical_=0

        #stability detection boundaries
        self.bn_max=1.8e-5
        self.bn_min=-1.8e-5
        # Element mass matrix (for 1D beam element)
        self.Me = (self.rho * self.A * self.le / 420.0) * np.array([
            [156,              22*self.le,     54,              -13*self.le],
            [22*self.le,       4*self.le**2,   13*self.le,      -3*self.le**2],
            [54,               13*self.le,     156,             -22*self.le],
            [-13*self.le,      -3*self.le**2,  -22*self.le,     4*self.le**2]
        ])

        # Stiffness matrices ( K = Ke - P KG )

        # Element linear stiffness matrix
        self.Ke = (self.E * self.I / self.le**3) * np.array([
            [12,           6*self.le,     -12,           6*self.le],
            [6*self.le,    4*self.le**2,  -6*self.le,    2*self.le**2],
            [-12,          -6*self.le,    12,            -6*self.le],
            [6*self.le,    2*self.le**2,  -6*self.le,    4*self.le**2]
        ])

        # Element geometric stiffness matrix (unit axial load)
        self.KGe = (1.0 / (30.0 * self.le)) * np.array([
            [36,            3*self.le,      -36,            3*self.le],
            [3*self.le,     4*self.le**2,   -3*self.le,     -self.le**2],
            [-36,           -3*self.le,     36,             -3*self.le],
            [3*self.le,     -self.le**2,    -3*self.le,     4*self.le**2]
        ])


        # Global matrices
        self.M = np.zeros((self.DOF, self.DOF))
        self.K = np.zeros((self.DOF, self.DOF))
        self.KG = np.zeros((self.DOF, self.DOF))
        # The damping matrix will be set to none untill we calculate the mass and the stiffness matrices because it is based on them
        self.C = None

        # Reduced matrices (After applying boundary conditions, the matrices will be reduced)
        self.Mr = None
        self.Kr = None
        self.KGr = None
        self.Cr = None




    # A function to assemble all the matrices of the elements (Mass , linear stiffness and geometric stiffness matrices)
    def Stiffness_assembler(self):
        # Reset before assembly
        self.M.fill(0.0)
        self.K.fill(0.0)
        self.KG.fill(0.0)

        for e in range(self.n):
            idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i in range(4):
                for j in range(4):
                    self.K[idx[i], idx[j]] += self.Ke[i, j]
                    self.KG[idx[i], idx[j]] += self.KGe[i, j]
                    self.M[idx[i], idx[j]] += self.Me[i, j]

        # the total damping matrix
        self.C = self.alpha * self.M + self.beta_rayleigh * self.K




    # A function to apply boundary conditions (reduce the system)
    def apply_BC(self):
        #for pinned-pinned U0 = 0 and Un = 0
        fixed_dofs = [0,self.DOF-2]  
        #calculate all indices that are not affected by the boundary conditions (for now, we only restrain params to zero)
        free_dofs = [i for i in range(self.DOF) if i not in fixed_dofs]

        # Reduce the matrices
        self.Kr = self.K[np.ix_(free_dofs, free_dofs)]
        self.KGr = self.KG[np.ix_(free_dofs, free_dofs)]
        self.Mr = self.M[np.ix_(free_dofs, free_dofs)]
        self.Cr = self.C[np.ix_(free_dofs, free_dofs)]

        # Save the new number of DOF for later uses
        self.free_dofs = free_dofs





    # We build the force vector f
    def build_force_vector(self,t,t_max):
        # The vector will be based on the new number of degrees of freedom
        n_red = len(self.free_dofs)
        F = np.zeros(n_red)

        # The perturbation lateral forces at specific points
        if (t/t_max)<0.02 or ((t/t_max)>0.2 and (t/t_max)<0.21) or ((t/t_max)>0.4 and (t/t_max)<0.41) or ((t/t_max)>0.6 and (t/t_max)<0.61) or ((t/t_max)>0.8 and (t/t_max)<0.81):
         F[201]=self.lateral_force
         #F[250]=self.lateral_force
        else:
         F[250]=0
         F[50]=0
        return F




    #The axial force which start from 0, increases untill it reaches the maximaume value at t_max - 1 , then it stays at maximaume fpr t = 1
    def axial_force_time(self, t, t_max):
        if (t/(t_max-0.5)<1):
         return self.axial_force*t/(t_max-0.5)
        else:
            return self.axial_force
        




    # A stability creteria, for now, it is set to detect any point of the beam passes two boundaries, this will be called each iteration
    def stability(self,u):
       # print(u[201])
        if np.any((u[201] > self.bn_max) | (u[201]< self.bn_min)):
            return 1
        else:
            return 2
                    



    # The solver 
    def Solver(self, t_max, t_step):

        # 1) Assemble and reduce system
        self.Stiffness_assembler()
        self.apply_BC()

        # 2) Reduced size
        n_red = self.Mr.shape[0]

        # 3) Time discretization
        self.n_steps = int(t_max / t_step) + 1
        # A vector of time steps
        time = np.linspace(0.0, t_max, self.n_steps)

        # 4) Newmark parameters
        Beta = 1.0 / 4.0
        gamma = 1.0 / 2.0

        c0 = 1.0 / (Beta * t_step**2)
        c1 = gamma / (Beta * t_step)
        c2 = 1.0 / (Beta * t_step)
        c3 = 1.0 / (2.0 * Beta) - 1.0
        c4 = gamma / Beta - 1.0
        c5 = t_step * (gamma / (2.0 * Beta) - 1.0)

        # 5) Allocate histories ( At each step we save data for later plotting)
        u = np.zeros((self.n_steps, n_red))
        v = np.zeros((self.n_steps, n_red))
        a = np.zeros((self.n_steps, n_red))

        # 6) initialize lateral load
        F_const = 0

        # 7) Initial acceleration
        P0 = self.axial_force_time(time[0], t_max)
        K0 = self.Kr - P0 * self.KGr

        a[0] = np.linalg.solve(
            self.Mr,
            F_const - self.Cr @ v[0] - K0 @ u[0]
        )

        # 8) Time integration loop
        for n in range(self.n_steps - 1):
            t_next = time[n + 1]
            

            # Current axial load
            P_next = self.axial_force_time(t_next, t_max)
            # Current lateral load
            F_const = self.build_force_vector(t_next, t_max)

            # Current tangent stiffness
            K_total = self.Kr - P_next * self.KGr

            # Effective stiffness
            K_eff = K_total + c0 * self.Mr + c1 * self.Cr
  
            # Effective force
            F_eff = (
                F_const
                + self.Mr @ (c0 * u[n] + c2 * v[n] + c3 * a[n])
                + self.Cr @ (c1 * u[n] + c4 * v[n] + c5 * a[n])
            )

            # Solve for displacement
            u[n + 1] = np.linalg.solve(K_eff, F_eff)

            # Update acceleration
            a[n + 1] = (
                c0 * (u[n + 1] - u[n])
                - c2 * v[n]
                - c3 * a[n]
            )
            
            # Update velocity
            v[n + 1] = (
                v[n]
                + t_step * ((1.0 - gamma) * a[n] + gamma * a[n + 1])
            )
            
            # Stability creteria
            if self.st == 0:
                val=self.stability(u[n + 1])
                if val ==1:
                    self.stable=False
                    print("nge3ret")
                    self.st=n + 1
                    self.cretical_=P_next
                    print(self.cretical_)
                    
                    
           
             
        return time, u, v, a

    




    # After solving, we need to plot, for that we need to reconstruct the original system because we reduce it after applying boundary conditions
    def reconstruct_full_displacement(self, u_reduced):
        n_steps = u_reduced.shape[0]
        u_full = np.zeros((n_steps, self.DOF))

        for i in range(n_steps):
            u_full[i, self.free_dofs] = u_reduced[i]

        return u_full





    def animate_beam(self, time, u, scale=1.0, speed=1):

        # reconstruct full DOFs
        u_full = self.reconstruct_full_displacement(u)

        # extract displacement only
        w = u_full[:, ::2]

        x = np.linspace(0, self.L, w.shape[1])

        fig, ax = plt.subplots()
        line, = ax.plot(x, w[0] * scale, 'b-', linewidth=2)
        ref_line1=ax.axhline(y=1.8e-5, color='black', linewidth=1.5)
        ref_line2=ax.axhline(y=-1.8e-5, color='black', linewidth=1.5)
        ax.set_xlim(0, self.L)
        status_text = ax.text(
            0.02, 0.95, "Stable", 
            transform=ax.transAxes, 
            fontsize=12,
            verticalalignment='top',
        
        )
        status_text.set_color('black')
        max_deflection = np.max(np.abs(w))
        ax.set_ylim(-1.2 * (max_deflection) * scale,
                    1.2 * (max_deflection) * scale)

        title = ax.set_title("")

        ax.set_xlabel("Beam length")
        ax.set_ylabel("Displacement")

        frame_indices = np.arange(0, len(time), speed)

        def update(frame):
            i = frame_indices[frame]
            if self.st != 0 and i >= self.st:
                ref_line1.set_color('red')
                ref_line2.set_color('red')
                status_text.set_color('red')
                status_text.set_text(f"UNSTABLE, critical load = {self.cretical_}")
            line.set_ydata(w[i] * scale)
            title.set_text(f"Time = {time[i]:.3f} s")
           
           
            fig.canvas.draw_idle()
            return line, title,  ref_line1, ref_line2

        ani = FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=30,
            blit=False
            )
        plt.grid()
        plt.show()

      
    def print_displacement_table(self, time, u):

        # reconstruct full displacement
        u_full = self.reconstruct_full_displacement(u)

        # positions
        positions = [0.25*self.L, 0.5*self.L, 0.75*self.L, 0.9*self.L]

        # nodes
        node_indices = [int(p / self.le) for p in positions]

        # DOFs (displacement only)
        dof_indices = [2*i for i in node_indices]

        # time sampling every 0.05 s
        dt = time[1] - time[0]
        step = int(0.05 / dt)
        time_indices = np.arange(0, len(time), step)

        # HEADER
        header = ["Time(s)"] + [f"x={p:.2f}m" for p in positions]
        print("\n" + "-"*60)
        print("{:<10} {:<12} {:<12} {:<12} {:<12}".format(*header))
        print("-"*60)

        # DATA
        for i in time_indices:
            row = [time[i]] + [u_full[i, dof] for dof in dof_indices]

            print("{:<10.3f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f}".format(*row))

        print("-"*60)
        


new_obj = Cantiliver_Beam_Solver(
    E=210e9,
    I=1.66e-9,
    L=1.0,
    n=400,
    A=0.20,
    rho=0.50
)

# Set the maximum axial load you want to ramp toward (3440N)
new_obj.axial_force =3700

time, u, v, a = new_obj.Solver(
    t_max=5.0,
    t_step=0.001,
)

new_obj.animate_beam(time, u, scale=1, speed=10)
new_obj.print_displacement_table(time, u)
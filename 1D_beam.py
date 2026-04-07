import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

class Cantiliver_Beam_Solver:
    # E: Yong modulus, I: moment of inertia, L:beam total length, n:number of elements
    def __init__(self,E,I,L,n):
        self.E=E
        self.I=I
        self.L=L
        self.n=n
        #element length
        self.le=L/n
        #number of nodes and their degree of freedom
        self.n_node=n+1
        self.node_DOF=2
        # total number of DOF
        self.DOF=self.node_DOF*self.n_node

        #Linear stiffness matrix 
        #for the element
        self.Ke = (self.E*self.I/self.le**3) * np.array([
        [12, 6*self.le, -12, 6*self.le],
         [6*self.le, 4*self.le**2, -6*self.le, 2*self.le**2],
         [-12, -6*self.le, 12, -6*self.le],
         [6*self.le, 2*self.le**2, -6*self.le, 4*self.le**2]
        ])
        #total linear stiffness
        self.K = np.zeros((self.DOF, self.DOF))

        #Geometric stiffness matrix
        #for the element
        self.KGe = (1/(30*self.le)) * np.array([
            [36, 3*self.le, -36, 3*self.le],
            [3*self.le, 4*self.le**2, -3*self.le, -self.le**2],
            [-36, -3*self.le, 36, -3*self.le],
            [3*self.le, -self.le**2, -3*self.le, 4*self.le**2]
        ])

        #total geometric stiffness
        self.KG = np.zeros((self.DOF, self.DOF))

        # Assemby of the stiffness matrices
    def Stiffness_assembler(self):
        for e in range(self.n):
         idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
         for i in range(4):
                for j in range(4):
                    self.K[idx[i], idx[j]] += self.Ke[i, j]
                    self.KG[idx[i], idx[j]] += self.KGe[i, j]


    #Applying boundary conditions (reducing the system)
    def apply_BC(self):
        fixed_dofs = [0, 1]
        
        self.K = np.delete(self.K, fixed_dofs, axis=0)
        self.K = np.delete(self.K, fixed_dofs, axis=1)
        
        self.KG = np.delete(self.KG, fixed_dofs, axis=0)
        self.KG = np.delete(self.KG, fixed_dofs, axis=1)

    # Solving the eigen_value problem
    def Solver(self):
        self.Stiffness_assembler()
        self.apply_BC()
        self.eigvals, self.eigvecs = eig(self.K, self.KG)

        return self.eigvals, self.eigvecs
    
    #Post proccessing of the results
    def Post_Proccessing(self,eigenvals,eigenvcs):
        # We take only the real part of the eigen values
        eigenvals = np.real(eigenvals)
        # We take only the positive eigen values
        eigenvals = eigenvals[eigenvals > 0]  
        #the smallest eigen value represents the cretical load of buckling
        self.cretical_load = np.min(eigenvals)
        print("The cretical load is :",self.cretical_load,"N")

        #Plotting
        #1-extract only displacements
        w=eigenvcs[::2]
        plt.figure(figsize=(10, 6))

        #Plot only the first 4 mode shapes
        for i in range(4):
            plt.plot(np.linspace(0, self.L, len(w[:, i])), w[:, i],label=f"Mode {i+1}")
        plt.title("Buckling Modes of Cantilever Beam")
        plt.xlabel("Beam length x")
        plt.ylabel("Deflection w(x)")
        plt.legend()
        plt.grid()
        plt.show()


 # E: Yong modulus, I: moment of inertia, L:beam total length, n:number of elements
new_obj = Cantiliver_Beam_Solver(210e9,1.66e-9,1,300)
results=new_obj.Solver()
new_obj.Post_Proccessing(results[0],results[1])

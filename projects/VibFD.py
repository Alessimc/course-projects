"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import sparse

t = sp.Symbol('t')

class VibSolver:
    """
    Solve vibration equation::

        u'' + w**2 u = f,

    """
    def __init__(self, Nt, T, w=0.35, I=1):
        """
        Parameters
        ----------
        Nt : int
            Number of time steps
        T : float
            End time
        I, w : float, optional
            Model parameters
        """
        self.I = I
        self.w = w
        self.T = T
        self.set_mesh(Nt)

    def set_mesh(self, Nt):
        """Create mesh of chose size

        Parameters
        ----------
        Nt : int
            Number of time steps
        """
        self.Nt = Nt
        self.dt = self.T/Nt
        self.t = np.linspace(0, self.T, Nt+1)

    def ue(self):
        """Return exact solution as sympy function
        """
        return self.I*sp.cos(self.w*t)

    def u_exact(self):
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return sp.lambdify(t, self.ue())(self.t)

    def l2_error(self):
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(self.dt*np.sum((ue-u)**2))

    def convergence_rates(self, m=4, N0=32):
        """
        Compute convergence rate

        Parameters
        ----------
        m : int
            The number of mesh sizes used
        N0 : int
            Initial mesh size

        Returns
        -------
        r : array_like
            The m-1 computed orders
        E : array_like
            The m computed errors
        dt : array_like
            The m time step sizes
        """
        E = []
        dt = []
        self.set_mesh(N0) # Set initial size of mesh
        for m in range(m):
            self.set_mesh(self.Nt+10)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [np.log(E[i-1]/E[i])/np.log(dt[i-1]/dt[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(dt)

    def test_order(self, m=5, N0=100, tol=0.1):
        r, E, dt = self.convergence_rates(m, N0)
        assert np.allclose(np.array(r[-1]), self.order, atol=tol), r

class VibHPL(VibSolver):
    """
    Second order accurate recursive solver

    Boundary conditions u(0)=I and u'(0)=0
    """
    order = 2

    def __call__(self):
        u = np.zeros(self.Nt+1)
        u[0] = self.I
        u[1] = u[0] - 0.5*self.dt**2*self.w**2*u[0]
        for n in range(1, self.Nt):
            u[n+1] = 2*u[n] - u[n-1] - self.dt**2*self.w**2*u[n]
        return u

class VibFD2(VibSolver):
    """
    Second order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2
    
    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        # want to solve matrix eq. Au = b as u= A^{-1} @ B
        u = np.zeros(self.Nt+1)

        # setting up the A matrix
        g = 2 - self.w**2 * self.dt**2
        
        A = sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -g), np.ones(self.Nt)], np.array([-1, 0, 1]), (self.Nt+1, self.Nt+1), 'lil')
        A[0,:2] = 1, 0
        A[-1,-2:] = 0, 1

        # setting up b vector with initial condition
        b = np.zeros(self.Nt+1)
        b[0] = self.I
        b[-1] = self.I

        # solve u= A^{-1} @ B
        A_csr = A.tocsr()
        u[:] = sparse.linalg.spsolve(A_csr, b)
        
        return u

class VibFD3(VibSolver):
    """
    Second order accurate solver using mixed Dirichlet and Neumann boundary
    conditions::

        u(0)=I and u'(T)=0

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        u = np.zeros(self.Nt+1)

        # setting up the A matrix
        g = 2 - self.w**2 * self.dt**2

        A = sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -g), np.ones(self.Nt)], np.array([-1, 0, 1]), (self.Nt+1, self.Nt+1), 'csr')
        A[0,:2] = 1, 0
        A[-1,-3:] = np.array([1, -4, 3]) / (2 * self.dt) # u'(T) = 0

        # setting up b vector with initial condition
        b = np.zeros(self.Nt+1)
        b[0] = self.I
        b[-1] = 0 # u'(T) = 0

        # solve u= A^{-1} @ B 
        u[:] = sparse.linalg.spsolve(A, b)
        return u

class VibFD4(VibFD2):
    """
    Fourth order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 4

    def __call__(self):
        u = np.zeros(self.Nt+1)

        # setting up the A matrix
        g = -30 + self.w**2 * 12 * self.dt**2

        supsub2 = - np.ones(self.Nt-1)
        supsub1 = np.full(self.Nt, 16)
        diag = np.full(self.Nt+1, g)

        
        A = sparse.diags([supsub2, supsub1, diag, supsub1, supsub2], np.array([-2, -1, 0, 1, 2]), (self.Nt+1, self.Nt+1), 'csr')

        A[0,:3] = 1, 0, 0
        A[-1,-3:] = 0, 0, 1

        # these need checking, might not need to change sign since no odd nr.
        A[1,:6] = np.array([10 / (12 * self.dt**2),
                            -15 / (12 * self.dt**2) + self.w**2 ,
                            -4 / (12 * self.dt**2),
                            14 / (12 * self.dt**2),
                            -6 / (12 * self.dt**2),
                            1 / (12 * self.dt**2)]) 
        
        A[-2,-6:] = np.array([1 / (12 * self.dt**2),
                             -6 / (12 * self.dt**2),
                             14 / (12 * self.dt**2),
                             -4 / (12 * self.dt**2),
                             -15 / (12 * self.dt**2) + self.w**2,
                             10 / (12 * self.dt**2)])

        # setting up b vector with initial condition
        b = np.zeros(self.Nt+1)
        b[0] = self.I
        b[-1] = self.I

        # solve u= A^{-1} @ B 
        u[:] = sparse.linalg.spsolve(A, b)
        return u

def test_order():
    w = 0.35
    VibHPL(8, 2*np.pi/w, w).test_order()
    VibFD2(8, 2*np.pi/w, w).test_order()
    VibFD3(8, 2*np.pi/w, w).test_order()
    VibFD4(8, 2*np.pi/w, w).test_order(N0=20)

# task 4
class VibFD2Extended(VibFD2):
    """
    Second order accurate solver for u'' + w^2 u = f(t) with Dirichlet boundary conditions
    """
    order = 2
    
    def __init__(self, Nt, T, w=0.35, I=1, ue=None):
        """
        ue : sympy expression, optional
            Exact solution for manufactured solution
        """
        super().__init__(Nt, T, w, I)
        self.ue_expr = ue
        if ue is not None:
            # Compute f(t) = u''(t) + w^2 u(t) symbolically using sympy
            self.f_expr = sp.diff(ue, t, 2) + self.w**2 * ue
            self.f = sp.lambdify(t, self.f_expr, "numpy")

    def ue(self):
        """Return exact solution as sympy function"""
        return self.ue_expr

    def u_exact(self):
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        if self.ue_expr is not None:
            ue_func = sp.lambdify(t, self.ue_expr, 'numpy')
            return ue_func(self.t)
        else:
            return np.zeros_like(self.t)  # Handle case when no exact solution is provided

    def __call__(self):
        u = np.zeros(self.Nt+1)

        # Compute f(t) at each time step
        f_vals = self.f(self.t) if self.ue_expr is not None else np.zeros(self.Nt+1)

        # Set up the A matrix as in the original VibFD2
        g = 2 - self.w**2 * self.dt**2
        A = sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -g), np.ones(self.Nt)], 
                         np.array([-1, 0, 1]), (self.Nt+1, self.Nt+1), 'csr')
        A[0, :2] = 1, 0  # Dirichlet boundary condition at t=0
        A[-1, -2:] = 0, 1  # Dirichlet boundary condition at t=T

        # Set up the right-hand side vector with f(t) included and ue(0) and ue(T)
        b = np.zeros(self.Nt+1)
        if self.ue_expr is not None:
            ue_func = sp.lambdify(t, self.ue_expr, 'numpy')
            b[0] = ue_func(0) # Dirichlet condition at t=0
            b[-1] = ue_func(self.T)  # Dirichlet condition at t=T

            b[1:-1] += f_vals[1:-1] * self.dt**2  # Include f(t) in the interior points
        else:
            b[0] = self.I
            b[-1] = self.I

        # Solve the system u = A^{-1} b
        u[:] = sparse.linalg.spsolve(A, b)

        return u



if __name__ == '__main__':
    test_order()
    w = 0.35
    # vib_solver = VibFD4(8, 2*np.pi/w, w)
    # solution = vib_solver()


    
    t = sp.Symbol('t')
    # testing task 4 for u(t) = t^4
    u_exact_t4 = t**4
    vib_solver_t4 = VibFD2Extended(Nt=8, T=2*np.pi/w, w=w, I=u_exact_t4.subs(t, 0), ue=u_exact_t4)
    vib_solver_t4.test_order()

    # testing task 4 for u(t) = exp(sin(t))
    u_exact_exp = sp.exp(sp.sin(t))
    vib_solver_exp = VibFD2Extended(Nt=8, T=2*np.pi/w, w=0.35, I=u_exact_exp.subs(t, 0), ue=u_exact_exp)
    vib_solver_exp.test_order() # not passing

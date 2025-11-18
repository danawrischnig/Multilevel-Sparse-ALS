from mpi4py import MPI
import numpy as np
from scipy.special import zeta
from dolfinx import fem, mesh
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
from typing import List


class Darcy1D:
    """
    Solves a 1D Darcy flow problem using the finite element method.

    Attributes:
        l (int): Mesh discretization level; number of elements is 2^l.
        d (int): Truncation order of the parametric diffusion coefficient.
        degree (int): Polynomial degree of Lagrange elements.
        y (List[fem.Constant]): List of symbolic parameter constants for the diffusion coefficient.
        problem (LinearProblem): Assembled variational problem to be solved.
    """

    def __init__(self, l: int = 3, d: int = 6, degree: int = 1) -> None:
        """
        Initializes the Darcy1D solver.

        Args:
            l (int): Mesh discretization level; number of elements is 2^l.
            d (int): Number of terms in the truncated diffusion coefficient expansion.
            degree (int): Degree of Lagrange basis functions.

        Raises:
            ValueError: If l is not between 1 and 15 (inclusive).
        """
        if not (1 <= l <= 15):
            raise ValueError(
                "Spatial discretization level must be between 1 and 15 (inclusive)"
            )

        self.l = l
        self.d = d
        self.degree = degree

        self._set_problem()

    def _set_problem(self) -> None:
        """
        Sets up the finite element problem:
            - creates mesh and function space
            - defines test/trial functions
            - constructs parametric diffusion coefficient
            - applies boundary conditions (robust for any degree + MPI)
            - defines the variational problem
        """
        # Create a 1D mesh on the interval [0, 1] with 2^l elements
        domain = mesh.create_interval(MPI.COMM_WORLD, 2**self.l, [0.0, 1.0])
        V = functionspace(mesh=domain, element=("Lagrange", self.degree))

        # Define test and trial functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Define symbolic parameters for the diffusion coefficient
        x = ufl.SpatialCoordinate(domain)[0]
        self.y: List[fem.Constant] = [
            fem.Constant(domain, PETSc.ScalarType(0)) for _ in range(self.d)
        ]

        # Base coefficient a_0
        delta = 0.01
        c = 1.01
        b = 0.75
        self.rho = c * np.arange(1, self.d + 1) ** b
        a = PETSc.ScalarType(delta + c * zeta(2 - b, 1))

        # Add truncated parametric sum to diffusion coefficient
        for j in range(self.d):
            a += self.y[j] / ((j + 1) ** 2) * ufl.cos((j + 1) * ufl.pi * x)

        # Right-hand side (constant source)
        f = fem.Constant(domain, PETSc.ScalarType(100.0))

        # Robust Dirichlet boundary condition u=0 at both ends (degree- and MPI-safe)
        tdim = domain.topology.dim
        facets = mesh.locate_entities_boundary(
            domain,
            tdim - 1,
            lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0),
        )
        dofs = fem.locate_dofs_topological(V, tdim - 1, facets)
        bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)

        # Assemble the linear problem
        self.problem = LinearProblem(
            a=a * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx,
            L=f * v * ufl.dx,
            bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

        # Keep references for later use
        self._domain = domain
        self._V = V

    def solve(self, y: np.ndarray) -> fem.Function:
        """
        Solves the variational problem for given parameter vector y.

        Args:
            y (np.ndarray): Array of shape (d,) for parametric diffusion coefficients.

        Returns:
            fem.Function: Solution u of the finite element problem.

        Raises:
            ValueError: If y does not have shape (d,).
        """
        y = np.asarray(y)
        if y.shape != (self.d,):
            raise ValueError(f"Parameter y must have shape {(self.d,)}, got {y.shape}")

        # Update parameter constants
        for val, const in zip(y, self.y):
            const.value = PETSc.ScalarType(val)

        # Solve the problem
        u = self.problem.solve()
        return u

    def get_integrated_solution(self, y: np.ndarray) -> float:
        """
        Computes the integral of the solution u over the domain.

        Args:
            y (np.ndarray): Coefficient vector for the diffusion term.

        Returns:
            float: ∫ u(x) dx over [0, 1] (globally reduced over MPI ranks).
        """
        u = self.solve(y)
        integral_form = fem.form(u * ufl.dx)
        I_local = fem.assemble_scalar(integral_form)
        I = self._domain.comm.allreduce(I_local, op=MPI.SUM)
        return float(I)


if __name__ == "__main__":
    # Example usage of the Darcy1D class
    l, d, N = 14, 6, 1000
    rng = np.random.default_rng(42)
    points = rng.uniform(-1, 1, (N, d))  # Random parameter vectors
    integrals1 = np.zeros(N)
    integrals2 = np.zeros(N)

    # Compare degree 1 vs 2 to verify consistency
    problem_p1 = Darcy1D(l=l, d=d, degree=1)
    problem_p2 = Darcy1D(l=l, d=d, degree=2)

    for i, y in enumerate(points):
        I1 = problem_p1.get_integrated_solution(y)
        I2 = problem_p2.get_integrated_solution(y)
        # They should be close; store the average just as an example
        integrals1[i] = I1
        integrals2[i] = I2

    print(f"mean squared difference: {np.mean((integrals1 - integrals2) ** 2)}")

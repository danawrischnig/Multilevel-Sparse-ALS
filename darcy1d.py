from mpi4py import MPI
import numpy as np
from scipy.special import zeta
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
from typing import List


class Darcy1D:
    """
    Solves a 1D Darcy flow problem using finite element discretization.

    Attributes:
        l (int): Level of spatial discretization, creating a mesh with 2^l sub-intervals.
        d (int): Number of terms used for truncating the diffusion coefficient sum.
        y (list): List of symbolic parameters in the diffusion coefficient.
        problem (LinearProblem): The finite element problem to be solved.
    """

    def __init__(self, l: int = 3, d: int = 6) -> None:
        """
        Initializes the Darcy1D solver with specified discretization level and truncation order.

        Args:
            l (int): Level of discretization, determining the number of mesh intervals (2^l).
            d (int): Truncation order for the diffusion coefficient expansion.

        Raises:
            ValueError: If l is not between 1 and 15.
        """
        if not (1 <= l <= 15):
            raise ValueError(
                "Spatial discretization level must be between 1 and 15 (inclusive)"
            )

        self.l = l  # Mesh discretization level
        self.d = d  # Truncation order for diffusion coefficient

        self._set_problem()

    def _set_problem(self) -> None:
        """
        Sets up the finite element problem, defining the mesh, function space,
        trial/test functions, diffusion coefficient, boundary conditions, and right-hand side.
        """
        # Create a 1D mesh with 2^l sub-intervals
        domain = mesh.create_interval(MPI.COMM_WORLD, 2**self.l, [0.0, 1.0])
        V = functionspace(mesh=domain, element=("Lagrange", 1))

        # Define test and trial functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Spatial coordinate and parameterized coefficients
        x = ufl.SpatialCoordinate(domain)[0]
        self.y: List[fem.Constant] = [
            fem.Constant(domain, PETSc.ScalarType(0)) for _ in range(self.d)
        ]

        # Define the diffusion coefficient as a truncated sum
        a = 1.01 * np.sqrt(2) * zeta(2, 1)  # Base coefficient a_0
        for j in range(self.d):
            trig_function = (
                ufl.cos((j // 2 + 1) * ufl.pi * x)
                if j % 2
                else ufl.sin((j // 2 + 1) * ufl.pi * x)
            )
            a += (self.y[j] / ((j // 2 + 1) ** 2)) * trig_function

        # Define right-hand side function
        f = fem.Constant(domain, default_scalar_type(100))

        # Define Dirichlet boundary conditions (u=0 at both ends of the domain)
        bc = fem.dirichletbc(
            value=PETSc.ScalarType(0),
            dofs=np.array([0, 2**self.l - 1], dtype=np.int32),
            V=V,
        )

        # Define the linear variational problem
        self.problem = LinearProblem(
            a=a * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx,
            L=f * v * ufl.dx,
            bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    def solve(self, y: np.ndarray) -> fem.Function:
        """
        Solves the Darcy flow problem for a given parameter vector y.

        Args:
            y (np.ndarray): Coefficient values for the diffusion term, with shape (d,).

        Returns:
            dolfinx.fem.Function: Solution function u of the finite element problem.

        Raises:
            ValueError: If the input y does not match the expected shape (d,).
        """
        y = np.asarray(y)
        if y.shape != (self.d,):
            raise ValueError(
                f"Parameter y must have dimension {(self.d,)}, got {y.shape}"
            )

        # Assign parameter values
        for val, constant in zip(y, self.y):
            constant.value = val

        # Solve the linear problem
        u = self.problem.solve()
        return u

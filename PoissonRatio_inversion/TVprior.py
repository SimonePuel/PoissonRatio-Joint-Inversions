'''
Created on Aug 27, 2021
Last modified on Jul 21, 2022

@authors: Simone Puel (spuel@utexas.edu) & Umberto Villa (uvilla@austin.utexas.edu)
'''

import dolfin as dl
import hippylib as hp


class vectorTVH1L2handler:
    """
    A linear combination of TV, H1 (i.e. L2 on the gradient), and L2 regularization.
    NOTE:
    - The weight of the L2 regularization (delta_val_L2) must be non-zero to ensure an invertible Hessian.
    - The parameter eps > 0 is a small perturbation needed to ensure the differentiability of the TV cost function
    """
    def __init__(self, gamma_val_TV, gamma_val_H1, delta_val_L2, eps=1.e-4):
        self.gamma_TV = dl.Constant( gamma_val_TV )
        self.gamma_H1 = dl.Constant( gamma_val_H1 )
        self.delta_L2 = dl.Constant( delta_val_L2 )
        self.eps      = dl.Constant( eps )

    def _w_TV(self, m):
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.eps )

    def __call__(self, m):
        return ( self.gamma_TV * self._w_TV(m) \
                 + self.gamma_H1 * dl.Constant(.5)*dl.inner( dl.grad(m), dl.grad(m) ) \
                 + self.delta_L2 * dl.Constant(.5)*dl.inner( m, m ) \
                )*dl.dx

    def gn_hessian(self, m, mtrial, mtest):
        return self.gamma_TV * dl.Constant(1.)/self._w_TV(m) * ( dl.inner( dl.grad(mtrial), dl.grad(mtest)) )*dl.dx \
             + self.gamma_H1 * dl.inner( dl.grad(mtrial), dl.grad(mtest) )*dl.dx \
             + self.delta_L2 * dl.inner( mtrial, mtest)*dl.dx



class NonGaussian_Prior( hp.modeling.prior._Prior ):
    """
    This class implements a general non-quadratice regularization model.

    Note: sampling is not supported for this class.


    Example usage:

    Vh_m: FE space for inversion parameter.
    mean: a dl.Vector for the prior mean (can be omitted if 0)
    gamma_val_TV, gamma_val_H1, delta_val_L2: Regularization parameters (must be floats)

    prior_handler = vectorTVH1L2handler(gamma_val_TV, gamma_val_H1, delta_val_L2)
    prior = NonGaussian_Prior(Vh_m, prior_handler, mean)


    """

    def __init__(self, Vh, prior_handler, mean=None, rel_tol=1e-12, max_iter=100):
        """
        Construct the total variation regularization model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:`gamma`:           parameter that controls regularization strength
        - :code:`eps`:             parameter that controls extra term for differentiability
        - :code:`mean`:            mean
        """
        self.Vh = Vh
        self.m = dl.Function(self.Vh)
        self.trial = dl.TrialFunction(Vh)
        self.test  = dl.TestFunction(Vh)

        self.rel_tol  = rel_tol
        self.max_iter = max_iter

        self.prior_handler = prior_handler

        # Mass matrix
        self.varfM = dl.inner(self.trial,self.test)*dl.dx
        self.M = dl.assemble(self.varfM)

        self.Msolver = hp.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = self.max_iter
        self.Msolver.parameters["relative_tolerance"] = self.rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False

        # Mean/initial guess
        self.mean = mean

        if self.mean is None:
            self.mean = dl.Vector(self.M.mpi_comm())
            self.init_vector(self.mean, 0)

        self.R = None

        self.Rsolver = hp.PETScLUSolver(self.Vh.mesh().mpi_comm(), "mumps")
#        self.Rsolver = hp.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "gmres", hp.amg_method("hypre_amg"))
#        self.Rsolver.parameters["maximum_iterations"] = self.max_iter
#        self.Rsolver.parameters["relative_tolerance"] = self.rel_tol
#        self.Rsolver.parameters["error_on_nonconvergence"] = True
#        self.Rsolver.parameters["nonzero_initial_guess"] = False

        self.setLinearizationPoint(self.mean, gauss_newton_approx=False)


    def cost(self, m):
        self.m.vector().zero()
        self.m.vector().axpy(1.0, m)
        self.m.vector().axpy(-1.0, self.mean)
        return dl.assemble(self.prior_handler(self.m))


    def grad(self, m, out):
        self.m.vector().zero()
        self.m.vector().axpy(1.0, m)
        self.m.vector().axpy(-1.0, self.mean)
        out.zero()
        grad_varf = dl.derivative(self.prior_handler(self.m), self.m, self.test)
        dl.assemble(grad_varf, tensor=out)


    def init_vector(self, x, dim):
        if dim == "noise":
            raise NotImplementedError("Method init_vector is not defined for TV regularization if dim == noise ")
        else:
            self.M.init_vector(x, dim)


    def sample(self, noise, s, add_mean=True):
        raise NotImplementedError("Method sample is not defined for TV regularization")


    def setLinearizationPoint(self, m, gauss_newton_approx=False):
        self.m.vector().zero()
        self.m.vector().axpy(1.0,m)
        self.m.vector().axpy(-1.0,self.mean)

        if gauss_newton_approx:
            h_varf    = self.prior_handler.gn_hessian(self.m, self.trial, self.test)
        else:
            grad_varf = dl.derivative(self.prior_handler(self.m), self.m, self.test)
            h_varf    = dl.derivative(grad_varf, self.m, self.trial)

        if self.R is None:
            self.R = dl.assemble(h_varf)
        else:
            self.R.zero()
            dl.assemble(h_varf, tensor=self.R)
        self.Rsolver.set_operator(self.R)

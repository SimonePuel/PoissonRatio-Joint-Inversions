"""
2D joint inversion for fault slip & shear modulus correlation parameters

This code estimates the correlation between the fault slip and the shear modulus by computing the forward problem with some perturbations:

\begin{equation}
s + \alpha s \quad\quad\quad\quad \mu + \beta \mu,
\end{equation}

and compute the data misfit $|| u - d ||_{\ell_2}$, where $d$ is the solution of the forward problem with $\alpha, \beta = 0$.


@ author: Simone Puel (spuel@utexas.edu)
"""



# Import libraries
import dolfin as dl
import ufl
import math
import numpy as np
# Import libraries for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=16)
rc('legend', fontsize=16)
TINY_SIZE = 14
SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 40
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

import sys
import os
#sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "...") )
import hippylib as hp
# Load TV regularization class and observation operator
from pointwiseStateObs import PointwiseStateObservation
from TVprior_Joint import *

# Make results reproducible
np.random.seed(seed=1)

# Set parameters compiler
dl.parameters["form_compiler"]["quadrature_degree"] = 5
dl.parameters["form_compiler"]["optimize"] = True
# Mute FFC and UFL warnings
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)



# Define functions
# Define the compliance matrix for elasticity
def AEsigma(s, mu, nu):
    A = 1./(2.*mu)*( s - nu/( 1 + nu*(dim-2) )*ufl.tr(s)*ufl.Identity(dim) )
    return A

# Define the asymmetry operator
def asym(s): 
    as_ = s[0,1] - s[1,0]
    return as_

# Define the tangent operator
def T(n):
    T_operator = ufl.as_vector( [n[1], -n[0]] )
    return T_operator

# Class to define different properties in the subdomains with anomaly built-in
class K_Triangle(dl.UserExpression):
    def __init__(self, subdomains, k_0, k_1, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.k_0 = k_0
        self.k_1 = k_1

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == triangle_surface:
            values[0] = self.k_0
        else:
            values[0] = self.k_1
    
    def value_shape(self):
        return ()
    


# Import mesh
path = "../mesh/"
name = "Mesh_CurvedFault_Triangle_PaperII"
mesh = dl.Mesh(path + name + '.xml')
xmin = -700.; xmax = 400.; ymin = -500.; ymax = 0.
# Extract dimension of the mesh
dim = mesh.topology().dim()
# Define normal component to boundaries
n = dl.FacetNormal(mesh)
# Define boundaries
boundaries = dl.MeshFunction("size_t", mesh, path + name + '_facet_region.xml')
subdomains = dl.MeshFunction("size_t", mesh, path + name + '_physical_region.xml')
top = 1
left = 2
bottom = 3
right = 4
fault = 5
blockleft = 7
blockright = 8
triangle = 9
triangle_surface = 10
ds = dl.Measure('ds')(domain=mesh, subdomain_data=boundaries)
dS = dl.Measure('dS')(domain=mesh, subdomain_data=boundaries)

# Define sep
sep = "\n"+"#"*80+"\n"




# Create function to interpolate in a region of interest (small portion of the domain)
def get_dx_for_error_evaluation(mesh, xmin, xmax, ymin, ymax):
    class LimitedDomain(dl.SubDomain):
        def inside(self, x, on_boundary):
            x_range = dl.between(x[0], (xmin, xmax))
            y_range = dl.between(x[1], (ymin, ymax))
            return x_range and y_range
    
    # Marker the subdomain
    marker_errors = dl.MeshFunction("size_t", mesh, dim)
    aroundFaultEdges = LimitedDomain()
    marker_errors.set_all(20)
    aroundFaultEdges.mark(marker_errors, 21)
    my_dx = dl.Measure("dx", domain=mesh, subdomain_data=marker_errors)
    
    return my_dx(21)



# The linear coseismic inversion problem
# Define the weak formulation of the FORWARD problem
def pde_varf(u, m, p):
    # Split the STATE and ADJOINT variables. Use dl.split() and not .split(deepcopy=True),
    # since the latter breaks FEniCS symbolic differentiation
    sigma, uu, r = dl.split(u)
    tau, w, q = dl.split(p)
    m_mu, m_s = dl.split(m)
    u0 = dl.Constant((0., 0.))
    # Define the weak formulation of the forward problem
    J = ufl.inner( AEsigma(sigma, m_mu, nu), tau )*ufl.dx \
        + ufl.inner( ufl.div(tau), uu )*ufl.dx \
        + ufl.inner( asym(tau), r )*ufl.dx \
        + ufl.inner( ufl.div(sigma), w )*ufl.dx + ufl.inner( asym(sigma), q )*ufl.dx \
        + ufl.inner( f, w )*dl.dx \
        - ufl.inner( u0, tau*n )*ds(bottom) \
        - ufl.inner( ufl.avg(m_s), ufl.dot(  T(n('+')), tau('+')*n('+') ) )*dS(fault)
    return J



# Perturbation theory: forward problem
def solveFwd(k, targets, alpha, beta, verbose=True):
    
    # Print values of perturbations: alpha_mu and beta_s
    print( "alpha_mu = %.3f while beta_s = %.3f" %(alpha, beta) )
    
    # Define function spaces
    # Use VectorFunctionSpace if the unknown is a vector field.
    # Use FunctionSpace object for scalar fields.
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)    # stress (tensor field, since BDM is a vector field)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)   # displacement (vector field)
    DGr = dl.FunctionSpace(mesh, "DG", k-1)         # rotation (scalar field)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    n = dl.FacetNormal(mesh)
    # Define mixed function spaces for the model parameters
    CG = dl.FunctionSpace(mesh, "CG", 1)
    CG_element = dl.MixedElement([CG.ufl_element(), CG.ufl_element()])
    Vm = dl.FunctionSpace(mesh, CG_element)
    # Combine the STATE, PARAMETER and ADJOINT function spaces
    Vh = [Vu, Vm, Vu]
    # Print the dofs of STATE, PARAMETER and ADJOINT variables
    ndofs = [ Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim() ]
    ndofs_state = [ Vu.sub(0).dim(), Vu.sub(1).dim(), Vu.sub(2).dim() ]
    if verbose:
        print( sep, "Set up the mesh and finite element spaces", sep )
        print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs) )
        print( "Number of STATE and ADJOINT dofs: STRESS={0}, DISPLACEMENT={1}, ROTATION={2}".format(*ndofs_state) )

    # Define the STATE and ADJOINT Dirichlet BCs
    zero_tensor = dl.Expression( (("0.", "0."), ("0.", "0.")), degree=0 )
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    
    # Define the true model PARAMETER
    mu_b = 39.5e9               # shear modulus background
    # Perturb shear modulus anomaly
    mu_a = (26.5 + alpha)*1e9   # shear modulus anomaly          
    print( "The shear modulus for the anomaly is: mu = %.2f GPa while the background mu = %.2f GPa" %(mu_a/1e9, mu_b/1e9) )
    
    # Expression for shear modulus
    mu_expr = K_Triangle(subdomains, mu_a, mu_b, degree=5)
    
    # Perturb fault slip
    slip = -10. + beta
    EQ_std = 15.
    EQ_depth = -20.
    s_expr = dl.Expression(('u0*exp(-(pow((x[1]-xc),2)/(pow(std,2))))'), u0=slip, xc=EQ_depth, std=EQ_std, degree=5)
    # Combine them
    m_mu_s = dl.Function(Vh[hp.PARAMETER]).vector()
    m_mu_ = dl.interpolate(mu_expr, CG).vector()
    m_s_ = dl.interpolate(s_expr, CG).vector()
    tmp = np.zeros(Vh[hp.PARAMETER].dim(),)
    tmp[0::2] = m_mu_.copy(); tmp[1::2] = m_s_.copy()                
    # Assign the values of the vector
    m_mu_s.set_local(tmp)
    m_mu_s.apply('')
    
    # Define the PDE problem
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Print the number of observations (observed surface horizontal and vertical displacements)
    if verbose:
        print( "Number of observation points: {0}".format(targets.shape[0]) )
    # Define the Misfit function to calculate the data misfit
    misfit = PointwiseStateObservation(Vh[hp.STATE], targets, indicator_vec=dl.interpolate(dl.Constant((0,0,0,0,1,1,0)), Vh[hp.STATE]).vector())
    
    # Solve FORWARD problem for the STATE variables & generate synthetic observations
    u_cal = pde.generate_state() # all dofs STATE variables and PETSC vector (not FEniCS Function)
    x = [u_cal, m_mu_s, None]
    pde.solveFwd(x[hp.STATE], x)    
    # Generate true observations by using the observation operator 'B'
    misfit.B.mult(x[hp.STATE], misfit.d)
    # Infer the index position of all non-zero entries in the misfit
    idx_d = list(np.nonzero(misfit.d)[0])              # misfit = 2*ntargets (2 displacement components) 
    # Extract horizontal and vertical displacement noisy observations
    d_obs = misfit.d[idx_d]
    
    return d_obs, mu_a/1e9, slip 



# Define a function to estimate the data misfit
def calculateDataMisfit(d_obs, d_ref):
    
    # Evaluate the data misfit w.r.t. reference
    d_misfit = np.linalg.norm((d_obs - d_ref), 2)
    print( "The data misfit is: d_misfit = %.6f" %(d_misfit) )
    
    return d_misfit




### DEFINE COMMON PARAMETERS ###
# Define order of elements (k = 1 --> linear; k = 2 --> quadratic)
k = 2
# Define body force
f = dl.Constant((0., 0.))
nu = 0.25

# Generate random surface observations
ntargets = 20
targets_x = np.linspace(-180., 0., ntargets) 
targets_y = (ymax-dl.DOLFIN_EPS)*np.ones(ntargets)
targets = np.zeros([ntargets, dim])
targets[:,0] = targets_x; targets[:,1] = targets_y



# Calculate reference observations at targets
alpha, beta = 0., 0.0

d_ref, mu_ref, s_ref = solveFwd(k, targets, alpha, beta, verbose=True)
print(d_ref)


# Decide if saving the results to a .txt file
save_txt = True

if save_txt:
    # Calculate data misfit over parameter space of perturbations: alpha_mu & beta_s
    alphas = np.linspace(-26, 70, 50)  # np.linspace(-5, 1, 10) # np.linspace(-26, 70, 50)
    betas  = np.linspace(-8, 8, 50)  # np.linspace(-8, 8, 20) # np.linspace(-8, 8, 50)
    # Initialize d_misfit, alpha_mu, beta_s
    data_misfits_ = []
    alpha_mu_ = []
    beta_s_ = []
    mus_ = []
    ss_ = []

    # Loop over perturbations
    for alpha in alphas:
        for beta in betas:
            # Append alpha and beta
            alpha_mu_.append(alpha)
            beta_s_.append(beta)

            # Evaluate the observed data at targets
            d_obs, mu, s = solveFwd(k, targets, alpha, beta, verbose=False)
            mus_.append(mu)
            ss_.append(s)
            # Estimate the L2-norm error w.r.t. reference observations
            d_misfit = calculateDataMisfit(d_obs, d_ref)

            # Append data misfit
            data_misfits_.append(d_misfit)

    # Convert lists into np.arrays
    mus = np.array(mus_)
    ss = np.array(ss_)
    alpha_mu = np.array(alpha_mu_)
    beta_s = np.array(beta_s_)
    data_misfits = np.array(data_misfits_) 



# Save the results to a .txt file
if save_txt:
    # Save alphas, betas, mus, slips, data misfits
    outFileName = '2D_Joint_Correlation_Analysis_Fwd_mu_nu.txt'
    csvoutput = open(outFileName, 'w+')
    # Save title line
    csvoutput.write( "alpha beta mu slip data_misfit\n" )
    # Save each line
    for j in range(len(alpha_mu)):
        csvoutput.write( "%.6f %.6f %.6f %.6f %.6f\n" %(alpha_mu[j], beta_s[j], mus[j], ss[j], data_misfits[j]) )
    csvoutput.close()





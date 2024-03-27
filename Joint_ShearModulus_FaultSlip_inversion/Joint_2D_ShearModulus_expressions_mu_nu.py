"""
This code solves several joint adjoint-based deterministic inversions 
by varying the shear modulus parametrization. This confirms how
robust are the results of the joint inversion regardless the choice of
the rigidity parametrization.

@ author: Simone Puel (spuel@utexas.edu)
"""


# Import libraries
import dolfin as dl
import ufl
import math
import numpy as np
import pandas as pd
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

import sys, os
#sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "...") )
import hippylib as hp
# Load TV regularization class, observation operator, and IO .xdmf
from pointwiseStateObs import PointwiseStateObservation
from TVprior_Joint import *
from FunctionsIO import FunctionsIO

# Make results reproducible
np.random.seed(seed=1)

# Set parameters compiler
dl.parameters["form_compiler"]["quadrature_degree"] = 5
dl.parameters["form_compiler"]["optimize"] = True
# Mute FFC and UFL dolfin warnings
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

# Define sep
sep = "\n"+"#"*80+"\n"



# Define the folder to save
meshpath = "../mesh/"
savepath = "./results_Joint_expression_mu_nu/"
os.makedirs(savepath, exist_ok=True)
GPa2Pa = 1e9
# Choose the end name of the mesh
endname = "_Triangle"


# Create a function that save the results for each test
def save_results(mu_expre, mesh, xf, yf, mu_true, m_mu, s_true, m_s, d_obs, d_cal):
    
    # Save results to HDF5
    namefolder = "fault_" + str(mu_expre) + endname + "/"
    path = savepath + namefolder

    # PARAMETER (mu_true, m_mu)
    filename = 'm_mu'
    function_names = ['true', 'inferred']
    function_list = [mu_true, m_mu]

    FunctionsIO.write(mesh.mpi_comm(), path + filename, function_names, function_list)

    # xf, yf
    fault = np.array([ xf, yf ]) 
    tosave = pd.DataFrame(fault.T, columns=['xf', 'yf'])
    filename = 'fault_geometry.txt'
    tosave.to_csv(path + filename, sep=' ', index=False)

    # slip
    s = np.array([ s_true, m_s ])
    tosave = pd.DataFrame(s.T, columns=['true', 'cal'])
    filename = 'm_s.txt'
    tosave.to_csv(path + filename, sep=' ', index=False)

    # observed data
    d = np.array([ d_obs, d_cal ])
    tosave = pd.DataFrame(d.T, columns=['obs', 'cal'])
    filename = 'data.txt'
    tosave.to_csv(path + filename, sep=' ', index=False)



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
class K(dl.UserExpression):
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
mesh = dl.Mesh(meshpath + name + '.xml')
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


# Define the weak formulation of the FORWARD problem
def pde_varf(u, m, p):
    # Split the STATE and ADJOINT variables. Use dl.split() and not .split(deepcopy=True),
    # since the latter breaks FEniCS symbolic differentiation
    sigma, uu, r = dl.split(u)
    tau, w, q = dl.split(p)
    m_mu, m_s = dl.split(m)
    u0 = dl.Constant((0., 0.))
    # Define the weak formulation of the forward problem
    mu = mu_expression(m_mu)
    J = ufl.inner( AEsigma(sigma, mu, nu), tau )*ufl.dx \
        + ufl.inner( ufl.div(tau), uu )*ufl.dx \
        + ufl.inner( asym(tau), r )*ufl.dx \
        + ufl.inner( ufl.div(sigma), w )*ufl.dx + ufl.inner( asym(sigma), q )*ufl.dx \
        + ufl.inner( f, w )*dl.dx \
        - ufl.inner( u0, tau*n )*ds(bottom) \
        - ufl.inner( ufl.avg(m_s), ufl.dot(  T(n('+')), tau('+')*n('+') ) )*dS(fault)
    return J


# Create a function to solve the joint deterministic adjoint-based inversion
def solveJointInversion(mesh, boundaries, k, targets, noise_std_dev, mtrue_mu_expr, mtrue_s_expr, mean_TV, m0_mu_s_expr, gamma_val_TV, gamma_val_H1_mu, gamma_val_H1_s, delta_val_L2, eps, generate_synthetics=False, verbose=True):
    
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
    mtrue_mu_s = dl.Function(Vh[hp.PARAMETER]).vector()
    mtrue_mu = dl.interpolate(mtrue_mu_expr, CG).vector()
    mtrue_s = dl.interpolate(mtrue_s_expr, CG).vector()
    tmp = np.zeros(Vh[hp.PARAMETER].dim(),)
    tmp[0::2] = mtrue_mu.copy(); tmp[1::2] = mtrue_s.copy()                
    # Assign the values of the vector
    mtrue_mu_s.set_local(tmp)
    mtrue_mu_s.apply('')
    # Define the starting model (initial guess) 'm_0'
    m0_mu_s = dl.interpolate(m0_mu_s_expr, Vh[hp.PARAMETER]).vector()

    # Define the PDE problem
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Print the number of observations (observed surface horizontal and vertical displacements)
    if verbose:
        print( "Number of observation points: {0}".format(targets.shape[0]) )
    # Define the Misfit function to calculate the data misfit
    misfit = PointwiseStateObservation(Vh[hp.STATE], targets, indicator_vec=dl.interpolate(dl.Constant((0,0,0,0,1,1,0)), Vh[hp.STATE]).vector())
    # Define the TV regularization components 
    reg_handler = vectorTVH1L2handler(gamma_val_TV, gamma_val_H1_mu, gamma_val_H1_s, delta_val_L2, eps=eps)
    mean = dl.interpolate( mean_TV, Vh[hp.PARAMETER] ).vector()
    reg = NonGaussian_Prior(Vh[hp.PARAMETER], reg_handler, mean=mean)
     
    if generate_synthetics: 
        # Solve FORWARD problem for the STATE variables
        utrue = pde.generate_state() 
        x = [utrue, mtrue_mu_s, None]
        pde.solveFwd(x[hp.STATE], x)
        # Generate true observations by using the observation operator 'B'
        misfit.B.mult(x[hp.STATE], misfit.d)
        idx_d = list(np.nonzero(misfit.d)[0])
        # Pollute true observations with random noise and generate synthetic data
        hp.Random().normal_perturb(noise_std_dev, misfit.d) # Comment if no data noise
        # Modify misift.d to remove the values for other variables except for displacement
        tmp = np.zeros(len(misfit.d),)
        tmp[idx_d] = misfit.d[idx_d].copy()                 # horizontal and vertical displacement misfit
        
    else:
        # Modify misift.d to remove the values for other variables except for displacement
        tmp = np.zeros(len(misfit.d),)
        # horizontal and vertical displacement misfit
        tmp[4::7] = d_observed[0::2]    # horizontal displacement misfit
        tmp[5::7] = d_observed[1::2]    # vertical displacement misfit
        
    # Assign the values of the new vector to misfit2.d
    misfit.d.set_local(tmp)
    misfit.d.apply('')
    misfit.noise_variance = noise_std_dev*noise_std_dev
    # Extract horizontal and vertical displacement noisy observations
    idx_d = list(np.nonzero(misfit.d)[0])
    d_obs = misfit.d[idx_d]    
    # Print the max displacement and the data noise std
    MAX = misfit.d.norm("linf")
    print( "The MAX displacement (m) is: %.4f and standard deviation of the noise (m) is: %.6f" %(MAX, noise_std_dev) )

    # Extract x,y coordinates of the fault for plotting
    bc1 = dl.DirichletBC(Vm, (10, 10), boundaries, fault)
    um = dl.Function(Vm)
    bc1.apply(um.vector())
    # Extract x,y coordinates of the fault and values
    xslip = dl.interpolate(dl.Expression( ("x[0]", "x[0]"), degree=5), Vm )
    yslip = dl.interpolate(dl.Expression( ("x[1]", "x[1]"), degree=5), Vm )
    xf_tot = xslip.vector()[um.vector() == 10] # x coordinate fault
    yf_tot = yslip.vector()[um.vector() == 10] # y coordinate fault
    xf = np.zeros( int(len(xf_tot)/2) )
    yf = np.zeros( int(len(xf_tot)/2) )
    for i in range( 0, len(xf) ):
        xf[i] = xf_tot[2*i]
        yf[i] = yf_tot[2*i+1]
    # Sort the values
    idx_f = np.argsort(xf)
    xf = xf[idx_f]; yf = yf[idx_f]
    
    # Construct the "Model" --> objective function
    model = hp.Model(pde, reg, misfit)
    
    ### CHECK the Gradient and Hessian with FINITE DIFFERENCE (FD) ###
    m = m0_mu_s.copy()
    # Use the hIPPYlib class to FD check
    # Plot the FD check.
    if False:
        _ = hp.modelVerify(model, m, is_quadratic=False)
        plt.show()

    if verbose:
        print( sep, "Solve the deterministic inverse problem", sep)
        
    # Solve the inverse problem
    solver = hp.ReducedSpaceNewtonCG(model)
    solver.parameters["rel_tolerance"]  = 1e-9
    solver.parameters["abs_tolerance"]  = 1e-12
    solver.parameters["max_iter"]       = 150 
    solver.parameters["GN_iter"]        = 5
    solver.parameters["globalization"]  = "LS"
    solver.parameters["LS"]["c_armijo"] = 1e-5
    solver.parameters["print_level"] = 0
    solver.parameters.showMe()

    x = solver.solve([None, m, None])

    if solver.converged:
        print( "\nConverged in ", solver.it, " iterations." )
    else:
        print( "\nNot Converged" )

    print( "Termination reason: ", solver.termination_reasons[solver.reason] )
    print( "Final gradient norm: ", solver.final_grad_norm )
    print( "Final cost: ", solver.final_cost )

         
    # Extract inverse solutions and transform to DOLFIN functions
    mtrue_fun = hp.vector2Function(mtrue_mu_s, Vh[hp.PARAMETER]) 
    mu_true_fun, s_true_fun = mtrue_fun.split(deepcopy=True)
    m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
    mu_fun, s_fun = m_fun.split(deepcopy=True)
    # Convert the "true" and "predicted" shear modulus to exact values (mu = exp(m_nu))
    mu_true = dl.project(mu_expression(mu_true_fun), CG )
    m_mu = dl.project(mu_expression(mu_fun), CG )
    # Print the range of the shear modulus solution
    if verbose:
        print( min(m_mu.vector()[:]/GPa2Pa), max(m_mu.vector()[:]/GPa2Pa) )
     
    # Extract values at the 1D fault (fault parallel)
    m_true_mu_s = mtrue_mu_s[um.vector() == 10]
    m_mu_s = m[um.vector() == 10]
    # Extract the slip
    m_true_s = np.zeros( int(len(m_true_mu_s)/2) )
    m_s_ = np.zeros( int(len(m_mu_s)/2) )
    for i in range( 0, len(m_true_s) ):
        m_true_s[i] = m_true_mu_s[2*i+1]
        m_s_[i] = m_mu_s[2*i+1]
    # Sort them
    s_true = m_true_s[idx_f[::1]]; s_true = np.flip(s_true) 
    m_s = m_s_[idx_f[::1]]; m_s = np.flip(m_s)
      
    # Extract the values for the data misfit = d_cal - d_obs. Use the
    # observational operator to extract the surface displacement: d_cal = Bu
    # Generate STATE and ADJOINT vectors
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    # Solve the forward problem to compute the calculated STATE variables
    model.solveFwd(u, x)
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    # Extract horizontal and vertical displacement predicted observations
    d_cal = misfit.Bu[idx_d]

    
    return( mesh, Vh, targets, xf, yf, idx_f, u, mu_true, m_mu, s_true, m_s, d_obs, d_cal )




### DEFINE COMMON PARAMETERS ###
# Define order of elements (k = 1 --> linear; k = 2 --> quadratic)
k = 2
# Define body force
f = dl.Constant((0., 0.))

# Define the true model PARAMETER
nu = 0.25
mtrue_s_expr = dl.Expression(('u0*exp(-(pow((x[1]-xc),2)/(pow(std,2))))'), u0=-10., xc=-20., std=15., degree=5)

# Generate random surface observations
ntargets = 20
targets_x = np.linspace(-180., 0., ntargets) 
targets_y = (ymax-dl.DOLFIN_EPS)*np.ones(ntargets)
targets = np.zeros([ntargets, dim])
targets[:,0] = targets_x; targets[:,1] = targets_y

# Choose the noise standard deviation
noise_std_dev = 0.002


# Exponential shear modulus parametrization
mu_expre = 'exp_m'
print( "Solving the joint inverse problem with: %s" %mu_expre )
###---------------------###
# Define the expression of shear modulus
def mu_expression(m):
    mu = ufl.exp(m)  
    return mu

# Define the shear modulus background and anomaly
mu_b = 24.4                # shear modulus background
mu_background = mu_expression(mu_b)
mu_a = 24.0                # shear modulus anomaly
mu_anomaly = mu_expression(mu_a) 
print( "The shear modulus for the anomaly is: mu = %.2f GPa while the background mu = %.2f GPa" %(mu_anomaly/1e9, mu_background/1e9) )
mtrue_mu_expr = K(subdomains, mu_a, mu_b, degree=5)
m0_mu_s_expr = dl.Expression( ( '23.5', '0.' ), degree=0 ) 
# Indicate the mean to get a consistent regularization
mean_TV = dl.Expression( ( '24.2198', '0.' ), degree=0 )

# Define regularization weights
eps = 1e-6 
gamma_val_TV = 1e-2 
gamma_val_H1_mu = 0. 
gamma_val_H1_s = 1e-2 
delta_val_L2 = 1e-8 

# Solve the inverse problem
( mesh, Vh, targets, xf, yf, idx_f, u, mu_true, m_mu, s_true, m_s, d_obs, d_cal ) = solveJointInversion(mesh, boundaries, k, targets, noise_std_dev, mtrue_mu_expr, mtrue_s_expr, mean_TV, m0_mu_s_expr, gamma_val_TV, gamma_val_H1_mu, gamma_val_H1_s, delta_val_L2, eps, generate_synthetics=True, verbose=True)
# Save results
save_results( mu_expre, mesh, xf, yf, mu_true, m_mu, s_true, m_s, d_obs, d_cal )


# Consider the exponetial parameterization as the reference true shear modulus structure for generating the synthetic data
# Save the synthetic data
mu_expre = 'exp_m'
loadfolder = "./results_Joint_expression_mu_nu/" + "fault_" + str(mu_expre) + endname + "/"
# This is the true range
print( "The shear modulus for the anomaly is: mu = %.2f GPa while the background mu = %.2f GPa" %(26.49, 39.52) )
# Tune parameters
mu_range = np.array([-0.54, 0.541])
mu0_ref = 33e9
a = 2.5
mu0 = mu0_ref / a
# Print the range of shear modulus
print( "The range is: %.2f - %.2f GPa" %(mu0*(a+np.tanh(mu_range[0]))/GPa2Pa, mu0*(a+np.tanh(mu_range[1]))/GPa2Pa ) )

# Import synthetic geodetic data d_obs
filename = "data.txt" 
d_obs_ = pd.read_csv(loadfolder + filename, delimiter=' ')
d_obs = d_obs_['obs'].to_numpy()

    

###---------------------###
# Shear modulus parametrization: alpha = 5
mu_expre = '33_5_tanh_m'
print( "Solving the joint inverse problem with: %s" % mu_expre )
# Define the expression of shear modulus
a = 5
def mu_expression(m):
    mu = (mu0_ref)/a*(a + ufl.tanh(m))  
    return mu

# Define the shear modulus background and anomaly
mu_b = 2.52               # shear modulus background
mu_background = mu_expression(mu_b) 
mu_a = -2.5               # shear modulus anomaly
mu_anomaly = mu_expression(mu_a)
print( "The shear modulus for the anomaly is: mu = %.2f GPa while the background mu = %.2f GPa" %(mu_anomaly/GPa2Pa, mu_background/GPa2Pa) )
mtrue_mu_expr_2 = K(subdomains, mu_a, mu_b, degree=5)
m0_mu_s_expr_2 = dl.Expression( ( '0.', '0.' ), degree=0 ) 
# Indicate the mean to get a consistent regularization
mean_TV_2 = dl.Expression( ( '0.', '0.' ), degree=0 ) 

# Define regularization weights
eps = 1e-7   
gamma_val_TV = 1e-1 
gamma_val_H1_mu = 0. 
gamma_val_H1_s = 2e-1 
delta_val_L2 = 1e-4 
d_observed = d_obs 
generate_synthetics = False

# Solve the inverse problem
( mesh, Vh, targets, xf, yf, idx_f, u_2, mu_true_2, m_mu_2, s_true, m_s_2, d_obs_2, d_cal_2 ) = solveJointInversion(mesh, boundaries, k, targets, noise_std_dev, mtrue_mu_expr_2, mtrue_s_expr, mean_TV_2, m0_mu_s_expr_2, gamma_val_TV, gamma_val_H1_mu, gamma_val_H1_s, delta_val_L2, eps, generate_synthetics=generate_synthetics, verbose=True) 
# Save results
save_results( mu_expre, mesh, xf, yf, mu_true_2, m_mu_2, s_true, m_s_2, d_obs_2, d_cal_2 )



###---------------------###
# Shear modulus parametrization: alpha = 10
mu_expre = '33_10_tanh_m'
print( "Solving the joint inverse problem with: %s" % mu_expre )
# Define the expression ofr shear modulus
a = 10
def mu_expression(m):
    mu = (mu0_ref)/a*(a + ufl.tanh(m))  
    return mu

# Define the shear modulus background and anomaly
mu_b = 2.5                # shear modulus background
mu_background = mu_expression(mu_b) 
mu_a = -2.5               # shear modulus anomaly
mu_anomaly = mu_expression(mu_a)
print( "The shear modulus for the anomaly is: mu = %.2f GPa while the background mu = %.2f GPa" %(mu_anomaly/GPa2Pa, mu_background/GPa2Pa) )
mtrue_mu_expr_3 = K(subdomains, mu_a, mu_b, degree=5)
m0_mu_s_expr_3 = dl.Expression( ( '0.', '0.' ), degree=0 ) 
# Indicate the mean to get a consistent regularization
mean_TV_3 = dl.Expression( ( '0.', '0.' ), degree=0 )

# Define regularization weights
eps = 1e-7 
gamma_val_TV = 1e1 
gamma_val_H1_mu = 0. 
gamma_val_H1_s = 1e2 
delta_val_L2 = 1e-5 
d_observed = d_obs 
generate_synthetics = False

# Solve the inverse problem
( mesh, Vh, targets, xf, yf, idx_f, u_3, mu_true_3, m_mu_3, s_true, m_s_3, d_obs_3, d_cal_3 ) = solveJointInversion(mesh, boundaries, k, targets, noise_std_dev, mtrue_mu_expr_3, mtrue_s_expr, mean_TV_3, m0_mu_s_expr_3, gamma_val_TV, gamma_val_H1_mu, gamma_val_H1_s, delta_val_L2, eps, generate_synthetics=generate_synthetics, verbose=True)
# Save results
save_results( mu_expre, mesh, xf, yf, mu_true_3, m_mu_3, s_true, m_s_3, d_obs_3, d_cal_3 )



###---------------------###
# Shear modulus parametrization: alpha = 2.5
mu_expre = '33_25_tanh_m'
print( "Solving the joint inverse problem with: %s" % mu_expre )
# Define the expression ofr shear modulus
a = 2.5
def mu_expression(m):
    mu = (mu0_ref)/a*(a + ufl.tanh(m))  
    return mu

# Define the shear modulus background and anomaly
mu_b = 0.541                # shear modulus background
mu_background = mu_expression(mu_b) 
mu_a = -0.54                # shear modulus anomaly
mu_anomaly = mu_expression(mu_a)
print( "The shear modulus for the anomaly is: mu = %.2f GPa while the background mu = %.2f GPa" %(mu_anomaly/1e9, mu_background/1e9) )
mtrue_mu_expr_4 = K(subdomains, mu_a, mu_b, degree=5)
m0_mu_s_expr_4 = dl.Expression( ( '0.', '0.' ), degree=0 ) 
# Indicate the mean to get a consistent regularization
mean_TV_4 = dl.Expression( ( '0.', '0.' ), degree=0 ) 

# Define regularization weights
eps = 1e-7   
gamma_val_TV = 1e-1  
gamma_val_H1_mu = 0. 
gamma_val_H1_s = 1e-1  
delta_val_L2 = 1e-4 
d_observed = d_obs 
generate_synthetics = False

# Solve the inverse problem
( mesh, Vh, targets, xf, yf, idx_f, u_4, mu_true_4, m_mu_4, s_true, m_s_4, d_obs_4, d_cal_4 ) = solveJointInversion(mesh, boundaries, k, targets, noise_std_dev, mtrue_mu_expr_4, mtrue_s_expr, mean_TV_4, m0_mu_s_expr_4, gamma_val_TV, gamma_val_H1_mu, gamma_val_H1_s, delta_val_L2, eps, generate_synthetics=generate_synthetics, verbose=True)
# Save results
save_results( mu_expre, mesh, xf, yf, mu_true_4, m_mu_4, s_true, m_s_4, d_obs_4, d_cal_4 )

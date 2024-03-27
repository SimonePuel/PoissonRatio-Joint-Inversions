# Adjoint-Based Geodetic Inversion: Simultaneous Estimation of Fault Slip and Heterogeneous Elastic Properties


This repository contains deterministic adjoint-based inversions of coseismic surface geodetic data for heterogeneous material elastic properties, including Poisson's ratio and shear modulus, as well as a joint inversion for both fault slip distribution and subduction zone structure. We utilized a second-order stress-accurate, stress-displacement-rotation mixed finite-element elastic formulation, which incorporates a new fault discontinuity implementation consistent with an infinite-dimensional formulation ([Puel et al., 2022](https://doi.org/10.1093/gji/ggac050)).

We used the open-source libraries, [FEniCS](https://fenicsproject.org) version 2019.1.0 (Logg \& Wells, 2010; Logg et al., 2012) and [hIPPYlib](https://hippylib.github.io) version 3.0.0 (Villa et al., 2016, 2018, 2021) for the forward and adjoint-based inverse modeling, respectively.


## Structure of the repository

In this repository, there are three folders and one file. The file contains [instructions](https://github.com/SimonePuel/PoissonRatio-Joint-Inversions/blob/main/FEniCS-hIPPYlib_installation.md) on how to install the open-source libraries. Meanwhile, the three folders contain files to reproduce the results in the paper by [Puel et al. (2024)](https://doi.org/10.1093/gji/ggad442):

- [PoissonRatio_inversion](https://github.com/SimonePuel/PoissonRatio-Joint-Inversions/tree/main/PoissonRatio_inversion): Contains the code to solve the deterministic adjoint-based inversion of surface geodetic data for the Poisson's ratio heterogeneous structure.
- [Joint_ShearModulus_FaultSlip_inversion](https://github.com/SimonePuel/PoissonRatio-Joint-Inversions/tree/main/Joint_ShearModulus_FaultSlip_inversion): Contains the code to solve the joint deterministic adjoint-based inversion of surface geodetic data for the shear modulus heterogeneous structure and fault slip distribution.
- [mesh](https://github.com/SimonePuel/PoissonRatio-Joint-Inversions/tree/main/mesh) contains the finite-element meshes used in the paper.


## Installation of FEniCS and hIPPYlib open-source libraries

For the installation of the two open-source libraries and all their dependencies, please refer to the provided file [FEniCS_hIPPYlib_installation.md](https://github.com/SimonePuel/PoissonRatio-Joint-Inversions/blob/main/FEniCS-hIPPYlib_installation.md).


## Mesh Generation and Visualization

For models with complex geometries, the open-source mesh geration software, [Gmsh](https://www.gmsh.info/) (Geuzaine \& Remacle, 2009), was used to create the finite-element meshes. [Matplotlib](https://matplotlib.org) was used to visualize the results in 2D.


## Please cite the following publications when using these codes 

We offer these codes freely, with the aim that they contribute to your research and educational endeavors. As part of standard scientific protocol, we kindly ask for acknowledgment of the authors' efforts by citing relevant peer-reviewed papers (listed below) in your presentations and publications. We are open to collaborative opportunities and encourage you to contact me at spuel@utexas.edu with any inquiries regarding the software or its potential applications.

- Simone Puel, Thorsten W. Becker, Umberto Villa, Omar Ghattas, Dunyu Liu (2024).An adjoint-based optimization method for jointly inverting heterogeneous material properties and fault slip from earthquake surface deformation data, _Geophysical Journal International_, **236**(2), pp. 778-797, DOI: https://doi.org/10.1093/gji/ggad442.

- Simone Puel, Eldar Khattatov, Umberto Villa, Dunyu Liu, Omar Ghattas, Thorsten W. Becker (2022). A mixed, unified forward/inverse framework for earthquake problems: fault implementation and coseismic slip estimate, _Geophysical Journal International_, **230**(2), pp. 733-758, DOI: https://doi.org/10.1093/gji/ggac050.

